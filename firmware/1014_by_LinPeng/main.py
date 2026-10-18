"""V12 BULLSEYE: OpenMV vision front-end for the KPZ-ESP32 car over UART3.

Deployment: open this file in OpenMV IDE and SAVE TO BOARD AS main.py.

WHAT V12 CHANGES (frame rate + robustness)
------------------------------------------
* FRAME RATE.  VGA + this detector measured 32 fps; QVGA reaches the sensor's
  46.6 fps ceiling and halves readout latency.  The camera now runs QVGA, all
  internal windows are QVGA, and the $G2,TGT report is scaled back to VGA
  units so the ESP32's pixel tuning is untouched.
* ADAPTIVE SEED BRIGHTNESS.  The seed used a fixed L*>=60 floor while this
  target's "10" disc measures L*~28..40 in a dim room -- nothing in the target
  could ever become a seed.  The floor now comes from the last accepted
  detection (lo + 0.62*(hi-lo)) with a short rotating fallback sweep.
* REAL GLYPH PROOF.  The "10" ink is verified by counting centre pixels that
  are clearly darker than the disc (target ~0.23, white crate 0.04, wall
  label 0.09).  It is now also applied while locked, which stops the tracker
  walking onto white walls when a frame is blurry.
* EXPOSURE.  Auto exposure measured ~57 ms in this room; a 57 ms frame smears
  a fast target over tens of pixels.  Brightness is normalised once at boot
  and moved into gain, capped at 10 ms exposure / 36 dB gain.
* TIGHTER LOCKED GATES + position jumps, Hough fallback off by default (it
  stalled the loop to ~1 fps and produced false circles).

WHAT V10 CHANGES (vs V9)
------------------------
* TARGET CHANGED.  V9 tracked a white disc with an ORANGE DOT in its centre.
  The target is now the printed silhouette target; the aim point is the centre
  of its WHITE "10" disc.  That disc carries a dark "10", is surrounded by the
  black 9-band, and has thin white concentric rings around it (the first at
  ~1.9 r).  The ORANGE-DOT gate is gone; the new identity proofs are:
    - dark annulus at 1.25 r (the 9-band),
    - a white concentric ring crossed between 1.5 r and 2.27 r,
    - the dark "10" glyph inside the disc on a COLD acquisition,
    - size/density/area gates, plus a locked-mode size gate that rejects the
      small white "5..9" numerals (they are ~4x smaller than the disc).
  The detector runs on a white-blob seed and validates geometry; no full-frame
  Hough is used, so the loop stays at ~40 fps on this OpenMV H7 at VGA.
* Everything else is V9: the ESP32 owns the gimbals, the Kalman tracker and
  predictive search window are unchanged, and the $G2 protocol is unchanged.

WHAT V10.1 FIXES (bench reports)
--------------------------------
* OpenMV IDE disconnects / board reboots mid-run: the 8 s hardware watchdog is
  now OFF by default (it reset the board whenever the IDE stopped the script),
  and the script no longer opens/reads USB_VCP (that stole the IDE's Ctrl-C /
  stop bytes).  Set ENABLE_WATCHDOG/USB_CONSOLE_ENABLE to True only when
  running untethered.  A disconnected USB can no longer kill the vision loop:
  print() is guarded and a vision error no longer calls tracker.off().
* Fast target loses lock: the search window is bigger, the prediction lead is
  only 15 ms (60 ms put the window ahead of the target), the locked geometry is
  relaxed for motion blur, and a LOCAL Hough circle fallback recovers the disc
  when blur merges its white blob with the white ring/numerals.  The locked
  gates now use the PREDICTED centre, not the clipped search-window centre.

The ESP32 owns the gimbals (GPIO27/14) and this script only *looks*.  This is
the V9 split: OpenMV reports target coordinates, the ESP32 runs the aiming PID
and drives the servos.

  * UART3 (P4=TX / P5=RX, 115200 8N1) is the link to the ESP32.
  * Every frame this script reports the vision result:

        $G2,TGT,<cx>,<cy>,<lx>,<ly>,<aim>,<run>!

      cx,cy   filtered target centre in image coords, -1,-1 = no target
      lx,ly   laser spot centre,                      -1,-1 = no laser
      aim     0 = centre the target on screen, 1 = bring the laser onto target
      run     1 = tracking, 0 = not tracking (the ESP32 must hold still)

  * It answers the ESP32's $G2 command frames well inside its 1 s timeout:

        PING    -> $G2,PONG,<seq>!
        INFO    -> $G2,ACK,<seq>,INFO,<7 ints>!
        LIMITS  -> $G2,ACK,<seq>,LIMITS,<4 ints>!
        others  -> $G2,ACK,<seq>,STATE,<text>!
        bad     -> $G2,ERR,<seq>,<reason>

  * When the laser locks onto the target it sends the unsolicited pair

        $G2,HIT,1!      (bumps the ESP32's hit counter)
        DONE\n          (lets the line-following car resume after parking)

  * It accepts PARKED\n from the ESP32 (car stopped at the parking marker).

  * It no longer touches P7/P8 at all.  Power-on auto-track is ON
    (AUTO_START = True).

The two-phase aiming *decision* (centre the target first, then switch the aim
reference to the laser spot) is unchanged, because it is a vision decision;
only the motion control lives on the ESP32.  That is why the `aim` field
exists: it tells the ESP32 which reference to servo towards.
"""
import math
import time
from pyb import UART

# ---------------------------------------------------------------- frame
# The camera runs QVGA (320x240).  Measured on this board that is the only
# way to reach the sensor's 46.6 fps ceiling: VGA with the same detector
# tops out at ~32 fps, QVGA at ~46 fps, and the 2x binning also halves
# readout latency and sensor noise.  All internal coordinates/windows are
# QVGA; the $G2,TGT report multiplies them by FRAME_SCALE so the ESP32
# keeps working in its existing VGA pixel tuning (deadband, PID, feed-
# forward are untouched).
FRAME_SCALE = 2

# ---------------------------------------------------------------- behaviour
AUTO_START = True    # Track on power-up without waiting for $G2,TRACK.
# Why True: if anything ever resets the board (or, with ENABLE_WATCHDOG on, the
# watchdog does), main.py starts again already tracking (run=1), so the ESP32 --
# whose mode gates whether the servos move -- resumes immediately.  With False
# it would come back in OFF, waiting for a $G2,TRACK that may never be re-sent.
# Set it back to False if you want the OpenMV to stay passive until commanded.
RUN_LIMITS = ((740, 2260), (1200, 1800))  # Recorded confirmed assembly travel, us.
BOOT_PAN_RESPONSE = 1  # Target-image dx sign for increasing pan PWM.
TILT_RESPONSE = 1      # Target-image dy sign for increasing tilt PWM.
SKIP_FRAMES_MS = 300
# ---- exposure / motion blur ------------------------------------------------
# Auto exposure measured ~57 ms in this room.  A 57 ms frame smears a moving
# target over tens of pixels -- that, not the tracker, is why a fast disc is
# lost.  Normalise brightness once at boot, then trade exposure for gain: same
# image brightness with 4-7x less motion blur.
MAX_EXPOSURE_US = 10000
MAX_GAIN_DB = 36.0       # measured sensor ceiling on this OV5640
SETTLE_MS = 1200         # let auto exposure/gain/AWB converge before locking
HOME_SETTLE_MS = 300
AUTO_STABLE_FRAMES = 3
TRACK_STABLE_FRAMES = 3
LOG_INTERVAL_MS = 1000
# Hardware watchdog: OFF by default.  With the watchdog running, stopping the
# script from the OpenMV IDE (or any pause longer than 8 s) leaves it unfed, so
# the board resets itself -- which looks exactly like "the IDE disconnected and
# the board rebooted mid-run".  Set True only for an untethered field run.
ENABLE_WATCHDOG = False
# Frames the target must stay centered (both axes in deadband) with a laser
# spot visible before the aim reference switches from image center to the
# laser spot (initial alignment lock).
ALIGN_LOCK_FRAMES = 5

# ------------------------------------------------------------------- bus
BUS_UART_ID = 3          # P4 = TX, P5 = RX on the OpenMV Cam H7 / H7 Plus
BUS_BAUD = 115200
BUS_READ_BUF = 256       # RX ring buffer, bytes
BUS_FRAME_MAX = 192      # same ceiling as the ESP32 side
BUS_TGT_MIN_MS = 20      # cap $G2,TGT at 50 Hz
BUS_SEND_HIT = True      # emit $G2,HIT,1! + DONE when the laser locks on
# The OpenMV IDE owns the same USB VCP.  If the script reads it, it steals the
# IDE's Ctrl-C/stop bytes and the IDE can lose the board; the custom `status` /
# `tgt` console is therefore OFF by default.  Telemetry via print() is
# unaffected and still shows in the IDE terminal.
USB_CONSOLE_ENABLE = False

# Once locked, only look near the previous position.  The scene contains other
# white objects with orange/yellow parts (a plate with fruit, an orange box,
# white walls/roll/shirt) and without this gate the detector happily switches to
# one of them the moment the real target leaves the frame, which makes the aim
# loop chase a phantom (observed as the reported target teleporting).
# When re-acquiring after a loss, how much an apparent disk-size mismatch counts
# against a candidate, in "pixels of distance".  Raising it trusts the
# remembered target size more than proximity to the last known position.
ROI_SIZE_WEIGHT = 1.0

# ---- target motion filter (alpha-beta) -------------------------------------
# Velocity is what makes a FAST target trackable: it feeds the feed-forward term
# on the ESP32 and lets us predict where the target will be by the time a servo
# command actually takes effect.  Without it the loop can only chase the last
# measured position and always arrives late.
# ---- target motion filter: 2-state Kalman, one per axis --------------------
# Replaces the old alpha-beta tracker.  Velocity is what drives the ESP32
# feed-forward *and* the search-window prediction, and alpha-beta needs many
# frames to converge on it -- a fast target cannot wait.  The Kalman filter
# converges in ~2 frames and also hands us the covariance, which is what lets
# the search window grow by measured uncertainty instead of by a fixed guess.
KF_ACCEL_STD = 750.0    # px/s^2: fast manoeuvres need a livelier filter
KF_MEAS_STD = 1.5        # px: detection noise on a good frame
KF_MAX_SPEED = 5000.0   # px/s sanity clamp (fast across-frame passes)
KF_MIN_DT = 0.001        # s

# ---- search window: predictive and adaptive --------------------------------
# Measured on this rig: a +-150px window costs ~4ms per frame, a FULL VGA scan
# ~42ms.  Going full-frame therefore halves the loop rate (46.6 -> ~21fps)
# exactly during the moments a fast target is hardest to hold.  So while a track
# is alive we never go full-frame: the window simply grows with the uncertainty.
# A fast target needs a bigger window and only a SMALL lead: the window is
# applied to the image that was just captured, so a 60 ms lead put the window
# 60 ms *ahead* of the target and a 6000 px/s target simply fell outside it.
ROI_BASE_HALF = 75      # px: minimum half-window
ROI_SIGMA_K = 6.0        # half-window += this * position sigma (px)
ROI_MAX_HALF = 170       # px: cap
ROI_DEAD_MS = 1000.0     # ms with no sighting before we give up and re-scan
ROI_LEAD_MS = 10.0       # ms: prediction lead for the window centre
ROI_LATENCY_S = 0.020    # s: velocity x timing spread added to the window

# ---- 锁定后的宽容判定 (V10 十号靶心) --------------------------------------
# 实测定位: "速度稍快就丢靶" 的主因不是舵机, 而是检出条件太苛刻 —— 冷启动那一套
# 要求所有几何条件在同一帧同时成立. 所以冷启动仍然严格(暗 "10" 字形 + 黑环 +
# 同心白环 + 尺寸), 一旦锁定就按"离上次位置的距离 + 尺寸相似度 + 分数"排序取最优,
# 不再要求全画面唯一; 运动模糊把圆盘拉扁/糊掉字形时也不会整帧丢弃.
# 目标换了: 不再是"白盘 + 橙点", 而是印刷靶的白色 10 环圆盘(黑色 "10" 字形、
# 外圈黑色 9 环、1.9r 处一圈细白环). 白字 "5..9" 比圆盘小约 4 倍, 锁定后由尺寸
# 门限挡住, 不会再被当成靶心.
#
# 丢失后保留"上次位置"作为弱参考的时长. 冷启动那套"全画面唯一"的规则在杂物环境里
# 几乎不可能满足, 于是"短暂丢失"就变成"永久丢失". 在这段时间内改为按"离丢失位置
# 的距离 + 尺寸相似度 + 分数"排序取最优, 才能真正重新捕获.
REACQ_MEMORY_MS = 5000.0
# 重捕窗口半径. 全画面扫描实测 ~40ms(20fps), 而有界窗口更小 —— 丢失重捕
# 正是最需要帧率的时候, 所以这段时间用有界窗口围着"上次位置"找, 而不是扫全画面.
REACQ_ROI_HALF = 140

# 每帧被拒原因计数 —— 用来量化"到底为什么丢", 每秒随日志打印一次
REJECT = {"size": 0, "span": 0, "contrast": 0, "glyph": 0, "dark": 0,
          "ring": 0, "jump": 0, "accept": 0}

# ---- coasting --------------------------------------------------------------
# The single biggest cause of "lost it at speed": one missed detection made the
# vision side report run=0 and the ESP32 freeze instantly, so the gimbal stopped
# dead and the target ran away.  Instead keep the track alive briefly, report the
# PREDICTED spot with run=1, and let the ESP32 keep moving.
TGT_COAST_MS = 500.0     # ms of prediction-only output after the last sighting

# --------------------------------------------------------------- vision
# TARGET: the "10" bullseye of the printed silhouette target.
#
# The aim point is a filled WHITE disc carrying a dark "10", surrounded by the
# BLACK 9-band, then thin WHITE concentric rings (the first at ~1.9 r).  The
# white "5..9" numerals are also white blobs, but they are about 4x smaller
# than the disc, so seeds are ranked by radius and, once locked, a candidate
# is only accepted when its size stays near the remembered disc size.  The
# proof that a blob is the bullseye is the combination:
#
#   1. the disc is a compact filled white blob (density/aspect/area gates);
#   2. the annulus at 1.25 r is black (the 9-band);
#   3. a white concentric ring is crossed between 1.5 r and 2.27 r;
#   4. on a COLD acquisition the disc centre also carries the dark "10" glyph
#      (a plain white card/plate has none), which is what keeps the very first
#      lock from grabbing a white object elsewhere in the room.
#
# This replaces the V9 white-disc + ORANGE-DOT target.  The orange-dot gate is
# gone (the new target has no dot); the dark "10" glyph, the dark annulus and
# the concentric ring are its identity proofs now.
#
# Brightness is measured in sRGB->L* units through a 256-entry lookup table.
# The tests are relative (interior vs annulus vs ring), so exposure changes do
# not need a retune, and the thresholds below are the ones verified live on the
# OpenMV H7 at VGA against this exact target.
WHITE = (60, 100, -40, 40, -40, 40)  # filled white 10 disc (neutral, L>=60)
# Red-laser gate.  NOTE: the old (70,100,66,100,5,55) was unsatisfiable --
# in L*a*b* a red bright enough to reach L*=70 cannot have A*=66, so no
# laser spot could ever match (the HUD always showed LASER -1,-1).
# Mid/bright red, strongly chromatic, not orange (A~37, B~75) and not
# yellow/white; size/shape/count gates reject red objects.
LASER = (50, 100, 45, 100, -20, 70)
LASER_MIN_PX = 2    # Tiny noise blobs do not count.
LASER_MAX_PX = 225
LASER_MAX_SIDE = 20
LASER_JUMP_PX = 50  # Spot is rigid with the camera; larger steps are noise.
AIM_ALPHA = 0.5
# The laser search is the second full-frame blob scan of every frame and costs
# about as much as the target search.  Restrict it to a window around the
# target: when the laser is doing its job it is ON the target, and this roughly
# doubles the achievable frame rate.  Set to 0 to scan the whole frame.
LASER_SEARCH_HALF = 55
ALIGNED_PX = 6
LIMITS = ((1460, 1540), (1480, 1550))
PROBE_LIMITS = ((500, 2500), (1100, 1900))
CENTER = (160, 120)
# Measured on this gimbal: the pan servo ignores commands below ~40-60 us and
# then jumps ~120-180 px, so a 12 px deadband asks for moves the hardware cannot
# make and the loop hunts forever.  Keep the deadband at the mechanical limit;
# lower it only if the linkage/servo resolution is improved.
DEADBAND_PX = 60
RESTART_PX = 75
GAIN_US_PER_SEC_PER_PX = 1.8
MAX_US_PER_SEC = 800      # Pan speed cap, us/s.
TILT_MAX_US_PER_SEC = 200  # Pitch speed cap, us/s.
TILT_GAIN_CAP = 1.2
ACCEL_US_PER_SEC2 = 2400

# ---- bullseye detector -----------------------------------------------------
# Seed: any compact filled white blob.  The disc is the big one; the white
# numerals and the thin rings are also white, so they are filtered by size,
# density and the geometric proofs below -- never by their absolute brightness.
SEED_MIN_PX = 20
SEED_MAX_FRAC = 0.08     # of the FULL frame (a white wall blob is much bigger)
SEED_MAX_SIDE = 110
SEED_ASPECT_MIN = 0.55
SEED_ASPECT_MAX = 1.50
SEED_DENSITY_MIN = 0.50
SEED_DENSITY_MAX = 0.92
SEED_CANDIDATES = 4      # cold scan; locked scan uses 2 (speed)
SEED_SIZE_WEIGHT = 0.025  # ranking bonus per pixel of blob radius

# Geometry of the target, in units of the disc radius r.
INTERIOR_SCALE = 0.55    # sample the white disc here
INTERIOR_PCT = 0.75      # "interior is bright" percentile of those samples
ANNULUS_SCALE = 1.25     # the black 9-band
ANNULUS_MIN = 0.60       # fraction of annulus samples at/below the midpoint
MIN_CONTRAST = 18        # interior L* must beat the annulus by at least this
MIN_CONTRAST_FRAC = 0.50 # ...and by this fraction of the disc brightness
RING_LO = 1.50           # white ring search: 1.50 r .. 2.27 r
RING_STEP = 0.085
RING_STEPS = 10
RING_FRAC = 0.50         # "bright ring" = annulus + this * (interior-annulus)
RING_MIN = 0.70          # of 16 rays that must cross the white ring
GLYPH_RADIUS_FRAC = 0.45  # central disc that holds the "10" glyph
GLYPH_DARK = 10           # a glyph pixel is this much darker than the disc
GLYPH_FRAC = 0.15         # cold: >= this share of the centre must be glyph ink
LOCK_GLYPH_FRAC = 0.06    # locked: blur softens the ink, keep a lower bar

# Candidate size sanity.  The disc's area should match pi r^2; the glyph hole
# leaves it just under 1.0.  Thin rings are far below this window.
BLOB_AREA_MIN = 0.55
BLOB_AREA_MAX = 1.15

# Cold acquisition must be the disc, not one of the small white numerals:
# the numerals are ~7..8 px in radius here and the disc is ~25 px.  Once the
# disc is locked the minimum drops so a far/small disc stays tracked.
COLD_MIN_R = 6
LOCKED_MIN_R = 4
LOCK_SIZE_MIN = 0.60     # locked: reject a candidate this much smaller...
LOCK_SIZE_MAX = 1.70     # ...or this much larger than the remembered disc

# Locked-mode motion-blur relaxation.  A target moving fast smears the disc and
# can blur/merge its edge with the white ring and numerals; the strict cold
# gates then reject the disc for one frame.  Once locked we only need positional
# continuity, so the geometry is relaxed -- but NOT so far that a white wall
# patch or a white crate passes: measured on this rig, the real disc scores
# contrast ~25, dark ~0.9, ring ~0.8, while the wall patches that used to steal
# the lock sit at contrast 10..15, dark ~0.5, ring ~0.2.
SEED_DENSITY_MIN_LOCK = 0.38
SEED_DENSITY_MAX_LOCK = 0.96
SEED_ASPECT_MIN_LOCK = 0.45
SEED_ASPECT_MAX_LOCK = 1.90
LOCK_CONTRAST = 16
LOCK_CONTRAST_FRAC = 0.40
LOCK_ANNULUS_MIN = 0.58
LOCK_RING_FRAC = 0.45
LOCK_RING_MIN = 0.35
LOCK_AREA_MIN = 0.50
LOCK_AREA_MAX = 1.22

# Locked-mode Hough fallback: when motion blur merges the white disc into the
# surrounding white ring/numerals, the blob seed disappears even though the
# disc edge is still a clear circle.  A LOCAL Hough in a small window around
# the prediction recovers the centre; this is the fast-motion safety net.
LOCK_HOUGH_THRESHOLD = 1500
HOUGH_X_MARGIN = 10
HOUGH_Y_MARGIN = 10
HOUGH_R_MARGIN = 10
# The Hough fallback is a last-resort fast-motion path.  A blurred disc can
# lose the thin white ring entirely, so it only requires the bright interior
# and the dark annulus; position + size gates still anchor identity.  These
# are tighter than V10.1 because a loose Hough gate locked onto a wall texture
# in every live run.
HOUGH_CONTRAST = 18
HOUGH_ANNULUS_MIN = 0.62
HOUGH_RING_MIN = 0.30
SIZE_WEIGHT = 0.015       # score bonus per pixel of radius (prefer the disc)
COLD_MARGIN = 0.12       # cold: best must beat 2nd by this score
SCORE_WEIGHT = 12.5      # px-equivalent of one score unit (locked ranking)

# ---- adaptive seed threshold ----------------------------------------------
# The root cause of "靶心识别不到": the seed used a FIXED L* floor of 60, but
# this printed target's "10" disc is only L*~28..40 in a dim room (the black
# 9-band is ~12).  Nothing in the target could ever become a seed, while white
# walls/crates (L*>=60) did.  Brightness is now taken from the target itself:
# every accepted detection reports its interior (hi) and annulus (lo), and the
# next seed threshold is lo + SEED_HINT_FRAC*(hi-lo) -- the measured midpoint
# between the black band and the disc.  On a cold start / after a long loss it
# sweeps a few absolute levels instead.
SEED_HINT_FRAC = 0.62
# Order matters: the first level is tried on the first cold frame.  With the
# exposure normalisation the disc usually lands at L* 38..70.
SEED_LEVELS = (40, 32, 48, 56, 28, 64)
SEED_TRY_LOCKED = 2      # thresholds tried per locked frame
_LAST_LEVEL = [None]     # seed L* hint from the last accepted detection
_LEVEL_PHASE = [0]       # rotates the fallback level when detection fails
# The local Hough fallback costs ~100 ms and, in live runs, stalled the loop to
# 1 fps exactly while the target was moving -- and its loose circle gate was a
# false-positive source.  With the 7x shorter exposure there is far less motion
# blur, so the blob path is enough.  Flip to True only if blur returns.
ENABLE_HOUGH = False

_COS = []
_SIN = []
for _k in range(16):
    _a = 2.0 * math.pi * _k / 16.0
    _COS.append(math.cos(_a))
    _SIN.append(math.sin(_a))
del _k, _a

# sRGB (8-bit) -> L* (0..100), so a pixel sampled with get_pixel() speaks the
# same units as the LAB percentiles.  Monotonic and cheap: one list lookup.
_LUT = []
for _i in range(256):
    _c = _i / 255.0
    if _c <= 0.04045:
        _y = _c / 12.92
    else:
        _y = ((_c + 0.055) / 1.055) ** 2.4
    if _y <= 0.008856:
        _l = 903.3 * _y
    else:
        _l = 116.0 * (_y ** (1.0 / 3.0)) - 16.0
    _LUT.append(int(_l + 0.5))
del _i, _c, _y, _l


def _l_at(img, x, y):
    """sRGB pixel at (x, y) as L* (0..100) through the LUT."""
    r, g, b = img.get_pixel(x, y)
    return _LUT[(r * 299 + g * 587 + b * 114) // 1000]


def _seed_score(blob, radius, density, prefer_size):
    """Rank seeds so the big filled disc comes before the numerals/rings."""
    score = SEED_SIZE_WEIGHT * radius - abs(density - 0.785) * 4.0
    if radius < 8.0:
        score -= 2.0
    if prefer_size:
        score -= abs(radius - 0.5 * prefer_size) / 8.0
    return score


def _seed_blobs(img, rx, ry, rw, rh, prefer_size, lmin=None):
    """Compact filled white blobs inside the search window, best first.

    `lmin` is the adaptive L* floor (see SEED_HINT_FRAC); when it is None the
    legacy fixed WHITE threshold is used.
    """
    locked = prefer_size is not None
    if locked:
        aspect_min = SEED_ASPECT_MIN_LOCK
        aspect_max = SEED_ASPECT_MAX_LOCK
        density_min = SEED_DENSITY_MIN_LOCK
        density_max = SEED_DENSITY_MAX_LOCK
    else:
        aspect_min = SEED_ASPECT_MIN
        aspect_max = SEED_ASPECT_MAX
        density_min = SEED_DENSITY_MIN
        density_max = SEED_DENSITY_MAX
    if lmin is None:
        thresholds = [WHITE]
    else:
        thresholds = [(int(lmin), WHITE[1], WHITE[2], WHITE[3],
                       WHITE[4], WHITE[5])]
    frame_cap = SEED_MAX_FRAC * img.width() * img.height()
    seeds = []
    try:
        blobs = img.find_blobs(thresholds, roi=(rx, ry, rw, rh),
                               pixels_threshold=SEED_MIN_PX,
                               area_threshold=SEED_MIN_PX, merge=False)
    except Exception:
        return seeds
    for blob in blobs:
        w = blob.w()
        h = blob.h()
        if w < 8 or h < 8:
            continue
        bx = blob.x()
        by = blob.y()
        # A seed cut by the search window has a wrong centre/size; the tracker
        # widens the window instead (same rule as V9).
        if (bx <= rx + 1 or by <= ry + 1
                or bx + w >= rx + rw - 2 or by + h >= ry + rh - 2):
            continue
        if (w if w > h else h) > SEED_MAX_SIDE:
            continue
        pixels = blob.pixels()
        if pixels > frame_cap:
            continue
        aspect = float(w) / h
        if aspect < aspect_min or aspect > aspect_max:
            continue
        density = blob.density()
        if density < density_min or density > density_max:
            continue
        radius = 0.5 * (w if w > h else h)
        seeds.append((blob, radius, density, pixels))
    seeds.sort(key=lambda s: _seed_score(s[0], s[1], s[2], prefer_size),
               reverse=True)
    return seeds


def _validate(img, blob, radius, density, pixels, require_glyph):
    """Return (score, hi, lo, dark, ring, density, cx, cy, size) or None."""
    width = img.width()
    height = img.height()
    cx = blob.cx()
    cy = blob.cy()
    min_r = COLD_MIN_R if require_glyph else LOCKED_MIN_R
    if radius < min_r:
        REJECT["size"] += 1
        return None
    if require_glyph:
        contrast_min = MIN_CONTRAST
        contrast_frac = MIN_CONTRAST_FRAC
        annulus_min = ANNULUS_MIN
        ring_frac = RING_FRAC
        ring_min = RING_MIN
        area_min = BLOB_AREA_MIN
        area_max = BLOB_AREA_MAX
    else:
        contrast_min = LOCK_CONTRAST
        contrast_frac = LOCK_CONTRAST_FRAC
        annulus_min = LOCK_ANNULUS_MIN
        ring_frac = LOCK_RING_FRAC
        ring_min = LOCK_RING_MIN
        area_min = LOCK_AREA_MIN
        area_max = LOCK_AREA_MAX

    interior = []
    annulus = []
    for k in range(16):
        c = _COS[k]
        s = _SIN[k]
        x = int(cx + INTERIOR_SCALE * radius * c + 0.5)
        y = int(cy + INTERIOR_SCALE * radius * s + 0.5)
        if 0 <= x < width and 0 <= y < height:
            interior.append(_l_at(img, x, y))
        x = int(cx + ANNULUS_SCALE * radius * c + 0.5)
        y = int(cy + ANNULUS_SCALE * radius * s + 0.5)
        if 0 <= x < width and 0 <= y < height:
            annulus.append(_l_at(img, x, y))
    if len(interior) < 12 or len(annulus) < 14:
        REJECT["span"] += 1
        return None
    interior.sort()
    annulus.sort()
    hi = interior[int(len(interior) * INTERIOR_PCT)]
    lo = annulus[len(annulus) // 2]
    # Absolute contrast (L* units) AND contrast normalised to the disc's own
    # brightness.  Measured: real target 1-lo/hi = 0.68, white crate/doorway
    # 0.28..0.47.  The absolute gate alone passed the scene edges because they
    # are high-contrast too; the ratio is what says "grey disc on black band".
    if (hi - lo) < contrast_min or (hi - lo) < contrast_frac * hi:
        REJECT["contrast"] += 1
        return None

    # 1.25 r must be the black 9-band.
    mid = (hi + lo) * 0.5
    dark = 0
    for value in annulus:
        if value <= mid:
            dark += 1
    dark_f = dark / float(len(annulus))
    if dark_f < annulus_min:
        REJECT["dark"] += 1
        return None

    # The disc carries the dark "10" glyph at its CENTRE.  Count pixels inside
    # a small disc (0.45 r) that are clearly darker than the disc interior.
    # Measured: printed target 0.23, white crate 0.04, wall label 0.09 -- this
    # replaced a p05/p60 histogram test that a noisy white box satisfied (its
    # square corners fell on the dark background).  It doubles as the identity
    # proof in LOCKED mode, which is what stops the tracker walking onto a
    # white wall patch when a frame is blurry.
    glyph_min = GLYPH_FRAC if require_glyph else LOCK_GLYPH_FRAC
    lim = int(radius * GLYPH_RADIUS_FRAC)
    if lim < 3:
        lim = 3
    step = int(radius / 8.0)
    if step < 1:
        step = 1
    dark = 0
    total = 0
    dy = -lim
    while dy <= lim:
        dx = -lim
        while dx <= lim:
            if dx * dx + dy * dy <= lim * lim:
                px = cx + dx
                py = cy + dy
                if 0 <= px < width and 0 <= py < height:
                    total += 1
                    if _l_at(img, px, py) < hi - GLYPH_DARK:
                        dark += 1
            dx += step
        dy += step
    if total >= 8 and (dark / float(total)) < glyph_min:
        REJECT["glyph"] += 1
        return None

    # 1.5 r .. 2.27 r: the thin white concentric ring.  A disc with no ring
    # (white plate, card, wall patch) fails here even if it is perfectly round.
    threshold = lo + ring_frac * (hi - lo)
    hits = 0
    for k in range(16):
        c = _COS[k]
        s = _SIN[k]
        found = False
        for t in range(RING_STEPS):
            rr = radius * (RING_LO + RING_STEP * t)
            x = int(cx + rr * c + 0.5)
            y = int(cy + rr * s + 0.5)
            if 0 <= x < width and 0 <= y < height:
                if _l_at(img, x, y) >= threshold:
                    found = True
                    break
        if found:
            hits += 1
    ring_f = hits / 16.0
    if ring_f < ring_min:
        REJECT["ring"] += 1
        return None

    area_ratio = pixels / (math.pi * radius * radius)
    if area_ratio < area_min or area_ratio > area_max:
        REJECT["size"] += 1
        return None

    score = ((hi - lo) / 100.0 + dark_f + 0.5 * ring_f
             + density + SIZE_WEIGHT * radius)
    return (score, hi, lo, dark_f, ring_f, density, cx, cy, int(2 * radius))


def _validate_center(img, cx, cy, radius):
    """Validate a Hough circle that has NO blob seed (motion-blur fallback).

    Same relative proofs as a locked blob, minus the area/glyph gates: bright
    interior, dark annulus at 1.25 r and the white concentric ring.  The radius
    comes from the circle fit, so a disc whose white blob merged with the white
    ring/numerals in a fast frame is still accepted.
    """
    width = img.width()
    height = img.height()
    cx = int(cx)
    cy = int(cy)
    if radius < LOCKED_MIN_R:
        REJECT["size"] += 1
        return None
    interior = []
    annulus = []
    for k in range(16):
        c = _COS[k]
        s = _SIN[k]
        x = int(cx + INTERIOR_SCALE * radius * c + 0.5)
        y = int(cy + INTERIOR_SCALE * radius * s + 0.5)
        if 0 <= x < width and 0 <= y < height:
            interior.append(_l_at(img, x, y))
        x = int(cx + ANNULUS_SCALE * radius * c + 0.5)
        y = int(cy + ANNULUS_SCALE * radius * s + 0.5)
        if 0 <= x < width and 0 <= y < height:
            annulus.append(_l_at(img, x, y))
    if len(interior) < 12 or len(annulus) < 14:
        REJECT["span"] += 1
        return None
    interior.sort()
    annulus.sort()
    hi = interior[int(len(interior) * INTERIOR_PCT)]
    lo = annulus[len(annulus) // 2]
    if (hi - lo) < HOUGH_CONTRAST:
        REJECT["contrast"] += 1
        return None
    mid = (hi + lo) * 0.5
    dark = 0
    for value in annulus:
        if value <= mid:
            dark += 1
    dark_f = dark / float(len(annulus))
    if dark_f < HOUGH_ANNULUS_MIN:
        REJECT["dark"] += 1
        return None
    threshold = lo + LOCK_RING_FRAC * (hi - lo)
    hits = 0
    for k in range(16):
        c = _COS[k]
        s = _SIN[k]
        found = False
        for t in range(RING_STEPS):
            rr = radius * (RING_LO + RING_STEP * t)
            x = int(cx + rr * c + 0.5)
            y = int(cy + rr * s + 0.5)
            if 0 <= x < width and 0 <= y < height:
                if _l_at(img, x, y) >= threshold:
                    found = True
                    break
        if found:
            hits += 1
    ring_f = hits / 16.0
    if ring_f < HOUGH_RING_MIN:
        REJECT["ring"] += 1
        return None
    score = ((hi - lo) / 100.0 + dark_f + 0.5 * ring_f
             + 0.785 + SIZE_WEIGHT * radius)
    return (score, hi, lo, dark_f, ring_f, 0.785, cx, cy, int(2 * radius))


def _hough_candidates(img, center_x, center_y, prefer_size):
    """Locked-only fallback: find the disc circle in a small window.

    Only called when the white-blob seed found nothing.  The Hough window is
    centered on the tracker's predicted position (the search-window centre),
    not on the last measurement, because the target may have moved during a
    fast frame.
    """
    half = int(prefer_size * 1.5)
    x0 = int(max(0, center_x - half))
    y0 = int(max(0, center_y - half))
    x1 = int(min(img.width(), center_x + half))
    y1 = int(min(img.height(), center_y + half))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return []
    r0 = 0.5 * prefer_size
    r_min = int(r0 * 0.60)
    if r_min < 8:
        r_min = 8
    r_max = int(r0 * 1.50)
    if r_max <= r_min + 3:
        r_max = r_min + 4
    try:
        circles = img.find_circles(roi=(x0, y0, x1 - x0, y1 - y0),
                                   threshold=LOCK_HOUGH_THRESHOLD,
                                   x_margin=HOUGH_X_MARGIN,
                                   y_margin=HOUGH_Y_MARGIN,
                                   r_margin=HOUGH_R_MARGIN,
                                   r_min=r_min, r_max=r_max, r_step=2)
    except Exception:
        return []
    out = []
    seen = {}
    for circle in circles:
        x = circle.x()
        y = circle.y()
        r = circle.r()
        key = (int(x) // 6, int(y) // 6, int(r) // 6)
        if key in seen:
            continue
        seen[key] = 1
        result = _validate_center(img, x, y, r)
        if result is not None:
            out.append(result)
    return out


def detect(img, search_roi=None, prefer=None, prefer_size=None):
    """Return (cx, cy, (x, y, w, h), ring_percent) of the 10-ring bullseye.

    Cold (prefer is None): strict -- dark glyph, dark annulus, white ring, and
    the best candidate must be unique by COLD_MARGIN, so the first lock cannot
    be a white plate/card/number.
    Locked (prefer given): identity comes from position and size.  Candidates
    far from the last sighting are rejected (unless their ring score is very
    strong), and the size must stay within LOCK_SIZE_MIN..LOCK_SIZE_MAX of the
    remembered disc, which is what stops the tracker switching to a white "9".
    """
    width = img.width()
    height = img.height()
    if search_roi is None:
        rx, ry, rw, rh = 0, 0, width, height
    else:
        rx, ry, rw, rh = search_roi
        if rx < 0:
            rw += rx
            rx = 0
        if ry < 0:
            rh += ry
            ry = 0
        if rx + rw > width:
            rw = width - rx
        if ry + rh > height:
            rh = height - ry
    if rw < 24 or rh < 24:
        return None

    locked = prefer is not None
    # Reference for the locked gates: the tracker's PREDICTED centre (passed
    # in as `prefer`), not the geometric centre of the search window.  When the
    # target is near a frame edge the search window is clipped and its centre
    # shifts by tens of pixels; using it made the tracker prefer the blob on
    # the clipped side instead of the true disc.
    if locked:
        refx = int(prefer[0])
        refy = int(prefer[1])
    else:
        refx = width // 2
        refy = height // 2

    # ---- adaptive seed thresholds ------------------------------------
    # Never sweep several full-frame scans in one frame: measured at VGA that
    # costs ~250 ms (4 fps) exactly while the car is trying to acquire.  Each
    # frame instead tries the level learned from the last accepted detection
    # (locked) plus ONE rotating fallback level.  A cold frame tries a single
    # level; if it misses, the next frame rotates.  Acquisition still completes
    # in a few frames while every frame stays fast.
    # NOTE: an Otsu histogram over a full VGA frame costs 37 ms here -- more
    # than the blob scan itself -- so the fallback is a fixed short sweep,
    # ordered by how often each level wins under the normalised exposure.
    hint = _LAST_LEVEL[0]
    levels = []
    if hint is not None:
        levels.append(hint)
        levels.append(hint + 10)
        levels.append(hint - 10)
    levels.extend(SEED_LEVELS)
    clean = []
    for value in levels:
        value = int(value)
        if value < 18:
            value = 18
        elif value > 85:
            value = 85
        if value not in clean:
            clean.append(value)
    levels = clean

    if locked and hint is not None:
        tries = [hint, levels[_LEVEL_PHASE[0] % len(levels)]]
    elif locked:
        phase = _LEVEL_PHASE[0]
        tries = [levels[phase % len(levels)],
                 levels[(phase + 1) % len(levels)]]
    else:
        tries = [levels[_LEVEL_PHASE[0] % len(levels)]]

    limit = 2 if locked else SEED_CANDIDATES
    require_glyph = not locked
    # Locked position gate.  The prediction already removes the target's
    # motion, so a large residual jump means "this is a different white
    # object".  V10.1 allowed 3r (~150 px) and the tracker walked from the
    # bullseye onto a white wall 240 px away.
    max_jump = 0
    far_jump = 0
    if locked:
        max_jump = int(3.0 * (prefer_size or 30))
        if max_jump < 40:
            max_jump = 40
        elif max_jump > 120:
            max_jump = 120
        far_jump = int(max_jump * 1.3)

    candidates = []
    used_level = None
    for lmin in tries:
        seeds = _seed_blobs(img, rx, ry, rw, rh, prefer_size, lmin)
        level_candidates = []
        for blob, radius, density, pixels in seeds[:limit]:
            result = _validate(img, blob, radius, density, pixels, require_glyph)
            if result is None:
                continue
            size = result[8]
            if prefer_size is not None:
                if not (LOCK_SIZE_MIN * prefer_size <= size
                        <= LOCK_SIZE_MAX * prefer_size):
                    REJECT["size"] += 1
                    continue
            if locked:
                dx = result[6] - refx
                dy = result[7] - refy
                jump_limit = far_jump if result[4] >= 0.55 else max_jump
                if dx * dx + dy * dy > jump_limit * jump_limit:
                    REJECT["jump"] += 1
                    continue
            level_candidates.append(result)
        if not level_candidates:
            continue
        level_candidates.sort(key=lambda c: c[0], reverse=True)
        if (not locked and len(level_candidates) > 1
                and (level_candidates[0][0] - level_candidates[1][0]) < COLD_MARGIN):
            # Ambiguous split at this level (disc vs. a white numeral/plate);
            # a different seed level can give a cleaner separation.
            continue
        candidates = level_candidates
        used_level = lmin
        break
    if candidates:
        _LEVEL_PHASE[0] = 0
    else:
        _LEVEL_PHASE[0] += 1

    # Fast-motion safety net: when the white disc blob merged with the white
    # ring/numerals, the seed either found nothing or produced an off-centre
    # merged blob.  Search locally around the PREDICTED position (the search
    # window centre), not around the last measurement, and let the score choose
    # between the blob and the Hough circle.
    need_hough = False
    if ENABLE_HOUGH and locked and prefer_size:
        if not candidates:
            need_hough = True
        else:
            jump_px = 0.30 * prefer_size
            jump2 = jump_px * jump_px
            for c in candidates:
                dx = c[6] - refx
                dy = c[7] - refy
                if (dx * dx + dy * dy) > jump2:
                    need_hough = True
                    break
    if need_hough:
        try:
            extra = _hough_candidates(img, refx, refy, prefer_size)
        except Exception:
            extra = []
        if extra:
            candidates = candidates + extra
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)

    if not locked:
        if len(candidates) > 1 and (candidates[0][0] - candidates[1][0]) < COLD_MARGIN:
            return None
        best = candidates[0]
    else:
        near = []
        for c in candidates:
            dx = c[6] - refx
            dy = c[7] - refy
            jump_limit = far_jump if c[4] >= 0.55 else max_jump
            if dx * dx + dy * dy <= jump_limit * jump_limit:
                near.append(c)
        if not near:
            return None
        if prefer_size:
            near.sort(key=lambda c: (
                ((c[6] - refx) ** 2 + (c[7] - refy) ** 2) ** 0.5
                + ROI_SIZE_WEIGHT * abs(c[8] - prefer_size)
                - SCORE_WEIGHT * c[0]))
        else:
            near.sort(key=lambda c: (
                ((c[6] - refx) ** 2 + (c[7] - refy) ** 2) ** 0.5
                - SCORE_WEIGHT * c[0]))
        best = near[0]

    REJECT["accept"] += 1
    # Remember the seed level that ACTUALLY isolated a valid disc.  Deriving it
    # from hi/lo instead (the old lo + 0.62*(hi-lo)) landed one level too low
    # for this target: the next frame then merged the disc into the black band
    # and dropped it.  Using the winning level makes the hand-off exact.
    level = used_level
    if level is None:
        level = int(best[2] + SEED_HINT_FRAC * (best[1] - best[2]))
    level = int(level)
    if level < 18:
        level = 18
    elif level > 85:
        level = 85
    _LAST_LEVEL[0] = level
    cx = best[6]
    cy = best[7]
    size = best[8]
    half = size // 2
    return (cx, cy, (cx - half, cy - half, size, size), best[4] * 100.0)

def valid_limits(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    for axis, pair in enumerate(value):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return False
        low, high = pair
        if type(low) is not int or type(high) is not int or low % 10 or high % 10:
            return False
        if not PROBE_LIMITS[axis][0] <= low <= LIMITS[axis][0]:
            return False
        if not LIMITS[axis][1] <= high <= PROBE_LIMITS[axis][1]:
            return False
    return True


def load_limits(path='v1_tracking_limits.json'):
    import json
    try:
        with open(path, 'r') as file:
            record = json.load(file)
        if record.get('version') == 1 and valid_limits(record.get('limits')):
            return record['limits']
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return RUN_LIMITS


def detect_laser(img, previous, search_roi=None):
    """Return the single valid laser spot (x, y) or None.

    Only a bright, VERY red, compact spot passes (LASER threshold), so
    orange objects and the orange target center cannot steal the aim
    reference. No target-center exclusion is applied here: when the laser
    is already ON the target its spot still counts, which keeps the gimbal
    stable and reports ALIGNED.

    search_roi limits the scan to a window (big speed win, see
    LASER_SEARCH_HALF); MicroPython rejects roi=None so fall back to a
    full-frame rect explicitly.
    """
    if search_roi is None:
        search_roi = (0, 0, img.width(), img.height())
    spots = []
    for spot in img.find_blobs([LASER], roi=search_roi, pixels_threshold=LASER_MIN_PX,
                               area_threshold=LASER_MIN_PX, merge=False):
        w, h = spot.w(), spot.h()
        if not (1 <= w <= LASER_MAX_SIDE and 1 <= h <= LASER_MAX_SIDE):
            continue
        if spot.pixels() > LASER_MAX_PX or spot.density() < 0.30:
            continue
        if not 0.4 <= w / h <= 2.5:
            continue
        spots.append((spot.cx(), spot.cy()))
    if len(spots) != 1:
        return None  # Ambiguous reflections are not a usable aim reference.
    x, y = spots[0]
    if previous and (abs(x - previous[0]) > LASER_JUMP_PX or abs(y - previous[1]) > LASER_JUMP_PX):
        return None  # The spot cannot jump; reject and keep the last reference.
    return x, y


class Track2D:
    """Two-state Kalman filter per axis: state = (position px, velocity px/s).

    Kept deliberately small (no matrices) so it stays cheap in MicroPython:
    every frame runs two of these, and the vision loop has ~21ms of budget.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = [0.0, 0.0]
        self.v = [0.0, 0.0]
        self.p00 = [0.0, 0.0]
        self.p01 = [0.0, 0.0]
        self.p11 = [0.0, 0.0]
        self.t = None

    def seed(self, px, py, now):
        self.x = [float(px), float(py)]
        self.v = [0.0, 0.0]
        for i in (0, 1):
            self.p00[i] = KF_MEAS_STD * KF_MEAS_STD
            self.p01[i] = 0.0
            self.p11[i] = KF_MAX_SPEED * KF_MAX_SPEED
        self.t = now

    def advance(self, now):
        """Roll the state forward to `now` with no measurement (coasting)."""
        if self.t is None:
            return False
        dt = time.ticks_diff(now, self.t) / 1000.0
        if dt <= KF_MIN_DT:
            return True
        q = KF_ACCEL_STD * KF_ACCEL_STD
        dt2 = dt * dt
        for i in (0, 1):
            self.x[i] += self.v[i] * dt
            self.p00[i] += 2.0 * dt * self.p01[i] + dt2 * self.p11[i] + q * dt2 * dt / 3.0
            self.p01[i] += dt * self.p11[i] + 0.5 * q * dt2
            self.p11[i] += q * dt
        self.t = now
        return True

    def correct(self, px, py, now):
        if self.t is None:
            self.seed(px, py, now)
            return
        self.advance(now)
        r = KF_MEAS_STD * KF_MEAS_STD
        for i, z in ((0, float(px)), (1, float(py))):
            a00 = self.p00[i]; a01 = self.p01[i]; a11 = self.p11[i]
            k0 = a00 / (a00 + r)
            k1 = a01 / (a00 + r)
            y = z - self.x[i]
            self.x[i] += k0 * y
            self.v[i] += k1 * y
            if self.v[i] > KF_MAX_SPEED:
                self.v[i] = KF_MAX_SPEED
            elif self.v[i] < -KF_MAX_SPEED:
                self.v[i] = -KF_MAX_SPEED
            self.p00[i] = (1.0 - k0) * a00
            self.p01[i] = (1.0 - k0) * a01
            self.p11[i] = a11 - k1 * a01

    def pos_at(self, now, lead_ms=0.0):
        """Where the target is (or will be) at `now` + lead; does not mutate."""
        if self.t is None:
            return None
        dt = time.ticks_diff(now, self.t) / 1000.0 + lead_ms / 1000.0
        if dt < 0.0:
            dt = 0.0
        return (self.x[0] + self.v[0] * dt, self.x[1] + self.v[1] * dt)

    def sigma(self):
        """Worst-axis 1-sigma position uncertainty in px."""
        a = self.p00[0] if self.p00[0] > 0.0 else 0.0
        b = self.p00[1] if self.p00[1] > 0.0 else 0.0
        if b > a:
            a = b
        return a ** 0.5


class Tracker:
    """Vision + aim-reference decision.  Owns no hardware.

    It no longer integrates servo motion (the ESP32 does that); it decides
    WHICH reference to aim at and publishes the result through `report`.
    """

    def __init__(self, limits=None):
        self.limits = [list(pair) for pair in (limits or LIMITS)]
        self.enabled = False          # logical "armed"; no PWM here
        self.pan_response = BOOT_PAN_RESPONSE
        self.mode = "OFF"
        self.reason = "BOOT"
        self.centered = [False, False]
        self.target = None
        self.laser = None
        self.aim = None               # Filtered laser spot; None => center mode.
        self.aim_error = (0.0, 0.0)
        self.stable = 0
        self._now = None              # last timestamp seen by update()
        self.kf = Track2D()
        self.last_sight_ms = None     # last frame that really saw the target
        self.filtered = None
        self.velocity = [0.0, 0.0]    # px/s, mirrored from the Kalman filter
        self.laser_aim = False        # True: aim reference is the laser spot.
        self.align_frames = 0         # Consecutive centered frames before lock.
        self.last_time = None
        self.report = None            # latest $G2,TGT frame (bytes)
        self.aligned_event = False    # True on the frame the laser locks on
        self.was_aligned = False
        self.last_seen = None         # last accepted detection, for the ROI gate
        self.last_size = None         # its disk width, for re-acquisition scoring
        self.miss = 0                 # consecutive frames with no target
        self.off()

    # ---------------------------------------------------------- state
    def off(self):
        self.enabled = False
        self.mode = "OFF"
        self.centered = [False, False]
        self.laser_aim = False
        self.align_frames = 0
        self.report = None
        self.last_seen = None
        self.last_size = None
        self.miss = 0
        self.kf.reset()
        self.last_sight_ms = None

    def predicted(self, now=None):
        """Predicted target centre for the detector's locked gates.

        `last_seen` is the last MEASUREMENT; on a fast target that is already
        one frame behind.  The detector's jump gate, its candidate sort and the
        Hough fallback should use the PREDICTED position instead.
        """
        if now is None:
            now = self._clock()
        if self.kf.t is None:
            return self.last_seen
        centre = self.kf.pos_at(now, ROI_LEAD_MS)
        if centre is None:
            return self.last_seen
        return (int(centre[0]), int(centre[1]))

    def search_roi(self, width, height, now=None):
        """Predictive, adaptive window; returns None only when the track is dead.

        It never falls back to a full-frame scan while a track is alive: measured
        cost is ~4ms for a +-150px window versus ~42ms for a full VGA frame, so a
        full-frame fallback halves the loop rate (46.6 -> ~21fps) exactly when a
        fast target is hardest to hold.  The window instead grows with the
        Kalman uncertainty and with how far the target travelled while blind.
        """
        if now is None:
            now = self._clock()
        if self.last_sight_ms is None:
            return None
        age = time.ticks_diff(now, self.last_sight_ms)
        if self.kf.t is None:
            # Track is dead but we may still hold a weak reference to where it
            # was lost.  Search a BOUNDED window around it: a full VGA scan costs
            # ~42ms (20fps) and re-acquisition is exactly when frame rate matters,
            # so a ~26ms window keeps us near 38fps while still covering 63% of
            # the frame right where the target most likely is.
            if self.last_seen is None or age > REACQ_MEMORY_MS:
                return None
            cx, cy = self.last_seen
            half = REACQ_ROI_HALF
            x0 = int(max(0, cx - half))
            y0 = int(max(0, cy - half))
            x1 = int(min(width, cx + half))
            y1 = int(min(height, cy + half))
            if x1 - x0 < 8 or y1 - y0 < 8:
                return None
            return (x0, y0, x1 - x0, y1 - y0)
        if age > ROI_DEAD_MS:
            return None
        centre = self.kf.pos_at(now, ROI_LEAD_MS)
        if centre is None:
            return None
        half = ROI_BASE_HALF + int(ROI_SIGMA_K * self.kf.sigma())
        # The prediction is made `ROI_LEAD_MS` ahead of the timestamp, while
        # the image itself was integrated over the exposure ending at `now`.
        # At 5000 px/s a 20 ms timing spread is 100 px, so the window must
        # carry velocity x latency or a fast target simply sits outside it.
        half += int(max(abs(self.kf.v[0]), abs(self.kf.v[1]))
                    * (age / 1000.0 + ROI_LATENCY_S))
        # Never make the window smaller than the disc itself, or a fast step
        # lands on the border and the clipped seed is thrown away.
        if self.last_size and half < self.last_size:
            half = self.last_size
        if half > ROI_MAX_HALF:
            half = ROI_MAX_HALF
        cx, cy = centre
        x0 = int(max(0, cx - half))
        y0 = int(max(0, cy - half))
        x1 = int(min(width, cx + half))
        y1 = int(min(height, cy + half))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def hold(self):
        self.mode = "HOLD"
        self.reason = "BUS_HOLD" if self.enabled else "USER"
        self.centered = [False, False]
        self.laser_aim = False
        self.align_frames = 0
        self.last_time = None

    def home(self):
        self.mode = "HOLD"
        self.reason = "HOME"
        self.centered = [False, False]
        self.laser_aim = False
        self.align_frames = 0

    def arm(self):
        if self.enabled:
            return False
        self.enabled = True
        self.mode = "HOLD"
        self.reason = "HOME"
        self.centered = [False, False]
        return True

    def begin_track(self, now):
        if self.mode != "HOLD":
            return False
        if not self.target or self.stable < TRACK_STABLE_FRAMES:
            return False
        self.mode = "TRACK"
        self.reason = "FOLLOW"
        self.last_time = now
        self.centered = [False, False]
        self.laser_aim = False
        self.align_frames = 0
        return True

    def bus_track(self):
        """$G2,TRACK / $G2,START: arm and start tracking ASAP."""
        self.enabled = True
        self.mode = "TRACK"
        self.reason = "BUS_TRACK"
        self.stable, self.target, self.filtered = 0, None, None
        self.centered = [False, False]
        self.laser_aim = False
        self.align_frames = 0
        self.last_time = None
        self.last_seen = None
        self.last_size = None
        self.miss = 0
        self.kf.reset()
        self.last_sight_ms = None

    # ---------------------------------------------------------- text
    def state_text(self):
        return "%s/%s" % (self.mode, self.reason)

    def status_line(self):
        x, y = self.target[:2] if self.target else (-1, -1)
        lx, ly = self.laser if self.laser else (-1, -1)
        return ("MODE %s REASON %s AIM %s TARGET %d,%d LASER %d,%d ERR %d,%d RUN %d" %
                (self.mode, self.reason, "LASER" if self.laser_aim else "CENTER",
                 x, y, lx, ly,
                 int(round(self.aim_error[0])), int(round(self.aim_error[1])),
                 1 if self.mode == "TRACK" else 0))

    # ---------------------------------------------------------- report
    def moving_allowed(self, now):
        """True when the ESP32 may drive the servos.

        Note it is NOT the same as mode == "TRACK": the auto-controller flips the
        mode to WAIT/WAIT_TARGET the instant one frame misses the target, and
        gating on TRACK there would defeat the whole point of coasting (the
        gimbal would freeze on every single dropped detection).  So: any live
        mode, except while a fresh acquisition is still being confirmed.
        """
        if self.mode in ("OFF", "BOOT", "HOLD"):
            return False
        if self.reason == "CONFIRM_TARGET":
            return False
        if self.last_sight_ms is None or self.kf.t is None:
            return False
        return time.ticks_diff(now, self.last_sight_ms) <= TGT_COAST_MS

    def build_report(self, now=None):
        """$G2,TGT,<cx>,<cy>,<vx>,<vy>,<lx>,<ly>,<aim>,<run>!

        cx/cy is the PREDICTED position at send time, not the last measurement:
        by the time the servos act the target has moved on, so the prediction is
        what should be aimed at.

        run=1 is held for TGT_COAST_MS after the last real sighting, so a single
        missed detection no longer makes the ESP32 freeze and lose the target.
        vx/vy feed the ESP32's feed-forward term.

        Internal coordinates are QVGA; everything published here is scaled to
        VGA units (FRAME_SCALE) so the ESP32's existing pixel tuning -- centre
        320,240, 120 px deadband, PID gains, velocity feed-forward -- is
        unchanged.
        """
        if now is None:
            now = self._clock()
        cx = cy = -1
        vx = vy = 0
        run = 0
        if self.moving_allowed(now):
            spot = self.kf.pos_at(now)
            if spot is not None:
                cx = int(round(spot[0])) * FRAME_SCALE
                cy = int(round(spot[1])) * FRAME_SCALE
                vx = int(round(self.kf.v[0])) * FRAME_SCALE
                vy = int(round(self.kf.v[1])) * FRAME_SCALE
                run = 1
        if self.laser:
            lx = int(self.laser[0]) * FRAME_SCALE
            ly = int(self.laser[1]) * FRAME_SCALE
        else:
            lx = ly = -1
        aim = 1 if (self.laser_aim and self.aim) else 0
        return ("$G2,TGT,%d,%d,%d,%d,%d,%d,%d,%d!"
                % (cx, cy, vx, vy, lx, ly, aim, run)).encode()

    # ---------------------------------------------------------- main step
    def _clock(self):
        """Frame timestamp, so the filter clock is testable and never stale."""
        return self._now if self._now is not None else time.ticks_ms()

    def update(self, target, laser, now):
        self._now = now
        dt = 0 if self.last_time is None else min(0.1, max(0, time.ticks_diff(now, self.last_time) / 1000))
        self.last_time = now
        previous = self.target
        self.target = target
        self.laser = laser
        if target:
            self.last_seen = (target[0], target[1])
            self.last_size = target[2][2]
            self.miss = 0
            # Keep a visible moving disk; reject discontinuous jumps/size changes.
            # Fast targets legitimately move more than 2 disc-widths between
            # frames; the old 160 px cap made every fast frame "discontinuous"
            # and reset the filter/stable counter.  Scale with the disc size.
            gate = max(80, min(320, 4 * max(target[2][2], target[2][3])))
            continuous = (previous and abs(target[0] - previous[0]) <= gate
                          and abs(target[1] - previous[1]) <= gate
                          and 0.5 <= target[2][2] / previous[2][2] <= 2
                          and 0.5 <= target[2][3] / previous[2][3] <= 2)
            self.stable = min(100, self.stable + 1) if continuous else 1
            if not continuous or self.kf.t is None:
                # New object (or first lock): restart the filter so a stale
                # velocity cannot be carried over onto a different target.
                self.kf.seed(target[0], target[1], now)
            else:
                self.kf.correct(target[0], target[1], now)
            self.last_sight_ms = now
        else:
            self.miss += 1
            # COAST instead of dropping the track.  A single missed detection used
            # to clear the filter immediately, which is exactly how a fast target
            # escaped: the aim loop lost its reference and the gimbal froze.
            # NOTE: the expiry test must NOT sit behind a kf.t guard.  Resetting
            # the filter clears kf.t, and if the expiry lived inside that guard
            # the weak reference could never expire -- it would pin detect() into
            # relaxed mode forever and eventually lock onto a decoy.
            if self.last_sight_ms is not None:
                age = time.ticks_diff(now, self.last_sight_ms)
                if age > ROI_DEAD_MS:
                    self.stable = 0
                    if self.kf.t is not None:
                        self.kf.reset()
                    # Keep last_seen/last_size as a WEAK reference while trying to
                    # re-acquire: detect() then ranks candidates by proximity and
                    # size instead of demanding a globally unique target, which
                    # in a cluttered room is effectively "lost forever".  It
                    # expires after REACQ_MEMORY_MS -> strict cold rules again.
                    if age > REACQ_MEMORY_MS:
                        self.last_seen = None
                        self.last_size = None
                        self.last_sight_ms = None
                elif self.kf.t is not None:
                    self.kf.advance(now)
        # Mirror the filter into the legacy fields the rest of the file uses.
        if self.kf.t is not None:
            self.filtered = [self.kf.x[0], self.kf.x[1]]
            self.velocity = [self.kf.v[0], self.kf.v[1]]
        else:
            self.filtered = None
            self.velocity = [0.0, 0.0]
        if laser is not None:
            if self.aim is None:
                self.aim = [float(laser[0]), float(laser[1])]
            else:
                for i in (0, 1):
                    self.aim[i] += AIM_ALPHA * (laser[i] - self.aim[i])
        else:
            # Laser lost: drop the old reference so we go back to centering
            # the target. (When the laser is ON the target it is still seen
            # by detect_laser, so this branch means the laser is truly gone.)
            self.aim = None

        self.aligned_event = False
        if self.mode not in ("TRACK", "WAIT"):
            self.was_aligned = False
            self.report = self.build_report()
            return
        if not target:
            # Target lost: WAIT in place, no scanning. Tracking resumes
            # automatically the moment a valid target reappears.
            self.mode = "WAIT"
            self.reason = "WAIT_TARGET"
            self.centered = [False, False]
            self.was_aligned = False
            self.report = self.build_report()
            return
        if self.stable < 2:
            self.mode = "WAIT"
            self.reason = "CONFIRM_TARGET"
            self.centered = [False, False]
            self.was_aligned = False
            self.report = self.build_report()
            return

        # Aim-reference selection (two-phase aiming):
        #   center mode (default): ref = image center -> initial alignment,
        #     the target is pulled to and kept at the screen center.
        #   laser mode: ref = laser spot -> the laser stays ON the target
        #     wherever the target moves; no more screen-centering.
        # Switch to laser mode only when a laser spot is visible AND the
        # target has stayed centered for ALIGN_LOCK_FRAMES frames.
        # Laser lost -> back to center mode immediately.
        if not self.laser_aim:
            if self.aim is not None and self.centered[0] and self.centered[1]:
                self.align_frames += 1
                if self.align_frames >= ALIGN_LOCK_FRAMES:
                    self.laser_aim = True
                    self.align_frames = 0
                    self.reason = "LASER_ALIGN"
            else:
                self.align_frames = 0
        else:
            if self.aim is None:
                self.laser_aim = False
                self.align_frames = 0
                self.reason = "CENTER_AGAIN"

        ref = self.aim if self.laser_aim and self.aim else (float(CENTER[0]), float(CENTER[1]))
        self.mode = "TRACK"
        self.reason = "FOLLOW"
        # Only the deadband decision is kept here: it drives both the laser-lock
        # state machine above and the reason reported to the ESP32.  The actual
        # servo motion is the ESP32's job.
        for axis in (0, 1):
            error = self.filtered[axis] - ref[axis]
            threshold = RESTART_PX if self.centered[axis] else DEADBAND_PX
            self.centered[axis] = abs(error) <= threshold
        self.aim_error = (self.filtered[0] - ref[0], self.filtered[1] - ref[1])
        aligned = max(abs(self.aim_error[0]), abs(self.aim_error[1])) <= ALIGNED_PX
        if self.laser_aim:
            self.reason = "ALIGNED" if aligned else "LASER_AIM"
        else:
            self.reason = "CENTERED" if aligned else "CENTER_TARGET"
        self.aligned_event = bool(aligned and self.laser_aim and not self.was_aligned)
        self.was_aligned = bool(aligned and self.laser_aim)
        self.report = self.build_report()


class AutoStart:
    """Power-on auto start (only used when AUTO_START = True)."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.enabled = False
        self.settle_at = None

    def cancel(self):
        self.enabled = False
        self.settle_at = None

    def start(self, now):
        t = self.tracker
        if t.mode in ("TRACK", "WAIT"):
            return None
        if not t.enabled and not t.arm():
            return "ARM_FAILED"
        self.enabled = True
        self.settle_at = now
        t.stable, t.target, t.filtered = 0, None, None
        t.reason = "AUTO_SETTLE"
        return None

    def update(self, now):
        if not self.enabled:
            return
        t = self.tracker
        if time.ticks_diff(now, self.settle_at) < HOME_SETTLE_MS:
            t.stable, t.target, t.filtered = 0, None, None
            return
        if t.mode != "HOLD":
            self.cancel()
            return
        t.reason = "AUTO_WAIT_TARGET"
        if t.target and t.stable >= AUTO_STABLE_FRAMES:
            if t.begin_track(now):
                self.cancel()


class Stats:
    def __init__(self):
        self.frames = 0
        self.cam_ok = 1
        self.cam_err = 0
        self.uart_err = 0
        self.snap_us = 0
        self.fps_x10 = 0
        self.t0 = time.ticks_ms()
        self.window = time.ticks_ms()
        self.window_frames = 0

    def frame_done(self, now, snap_us):
        self.frames += 1
        self.window_frames += 1
        self.snap_us = snap_us
        elapsed = time.ticks_diff(now, self.window)
        if elapsed >= 1000:
            self.fps_x10 = int(self.window_frames * 10000 / elapsed)
            self.window = now
            self.window_frames = 0

    def uptime_ms(self):
        return time.ticks_diff(time.ticks_ms(), self.t0)


class BusServer:
    """$G2 server on UART3: parse commands, answer, and stream $G2,TGT."""

    def __init__(self, uart, tracker, stats):
        self.uart = uart
        self.tracker = tracker
        self.stats = stats
        self.frame = bytearray()
        self.line = bytearray()
        self.parked = False
        self.last_cmd = ""
        self.tgt_sent = 0
        self.cmds = 0
        self.tx_bytes = 0

    # -------------------------------------------------------- low level
    def write(self, data):
        if self.uart is None:
            return False
        try:
            self.uart.write(data)
            self.tx_bytes += len(data)
            return True
        except OSError:
            self.stats.uart_err += 1
            return False

    def poll(self):
        if self.uart is None:
            return
        pending = self.uart.any()
        while pending:
            data = self.uart.read(pending)
            if not data:
                break
            for byte in data:
                self.feed(byte)
            pending = self.uart.any()

    def feed(self, byte):
        if byte == 0x24:                       # '$' starts a $G2 frame
            self.frame = bytearray(b'$')
            self.line = bytearray()
            return
        if self.frame:
            if len(self.frame) >= BUS_FRAME_MAX:
                self.frame = bytearray()
                self.stats.uart_err += 1
                return
            self.frame.append(byte)
            if byte == 0x21:                   # '!'
                frame = bytes(self.frame)
                self.frame = bytearray()
                self.handle_frame(frame)
            return
        # Bare, newline-terminated words (PARKED / DONE).
        if byte in (10, 13):
            if self.line:
                self.handle_line(bytes(self.line))
                self.line = bytearray()
            return
        if len(self.line) < 24:
            self.line.append(byte)

    # -------------------------------------------------------- dispatch
    def handle_frame(self, frame):
        if not frame.startswith(b'$G2,'):
            return
        body = frame[1:-1]                     # drop '$' and '!'
        parts = body.split(b',', 3)
        if len(parts) < 3:
            self.stats.uart_err += 1
            return
        try:
            verb = parts[1].decode()
        except UnicodeError:
            # line noise (e.g. while the ESP32 reboots) must not kill the script
            self.stats.uart_err += 1
            return
        self.cmds += 1
        self.dispatch(verb, parts[2], parts[3] if len(parts) > 3 else b'')

    def handle_line(self, line):
        text = line.strip().decode()
        if text == "PARKED":
            self.parked = True
            print("BUS: ESP32 reports PARKED (car stopped)")

    def ack(self, seq, kind, payload=b''):
        # The ESP32 expects the prefix "$G2,ACK,<seq>,STATE," for every command
        # except INFO and LIMITS, so `kind` must be STATE for all of those.
        self.write(b'$G2,ACK,' + seq + b',' + kind + b',' + payload + b'!')

    def dispatch(self, verb, seq, args):
        self.last_cmd = verb
        t = self.tracker
        if verb == "PING":
            self.write(b'$G2,PONG,' + seq + b'!')
            return
        if verb in ("START", "TRACK"):
            t.bus_track()
            self.ack(seq, b'STATE', t.state_text().encode())
        elif verb in ("HOLD", "STOP"):
            t.hold()
            self.ack(seq, b'STATE', t.state_text().encode())
        elif verb == "OFF":
            t.off()
            self.ack(seq, b'STATE', t.state_text().encode())
        elif verb == "HOME":
            t.home()
            self.ack(seq, b'STATE', t.state_text().encode())
        elif verb == "STATUS":
            self.ack(seq, b'STATE', t.state_text().encode())
        elif verb == "INFO":
            self.write(b'$G2,ACK,' + seq + b',INFO,' + self.info_payload() + b'!')
        elif verb == "LIMITS":
            self.write(b'$G2,ACK,' + seq + b',LIMITS,' + self.limits_payload() + b'!')
        elif verb in ("CALPAN", "JOG", "ACCEPT"):
            # The ESP32 owns the servos now, so these are meaningfully local to
            # it; we just acknowledge so its command queue never times out.
            self.ack(seq, b'STATE', b'LOCAL')
        else:
            self.write(b'$G2,ERR,' + seq + b',UNKNOWN_VERB!')

    # -------------------------------------------------------- payloads
    def info_payload(self):
        # uptime_ms, frames, fps_x10, snap_us, camera_ok(0/1), cam_err, uart_err
        return ("%d,%d,%d,%d,%d,%d,%d" % (
            self.stats.uptime_ms(), self.stats.frames, self.stats.fps_x10,
            self.stats.snap_us, self.stats.cam_ok, self.stats.cam_err,
            self.stats.uart_err)).encode()

    def limits_payload(self):
        lim = self.tracker.limits
        return ("%d,%d,%d,%d" % (lim[0][0], lim[0][1], lim[1][0], lim[1][1])).encode()

    # -------------------------------------------------------- telemetry
    def send_report(self):
        if self.tracker.report is None:
            return
        if self.write(self.tracker.report):
            self.tgt_sent += 1

    def send_hit_and_done(self):
        if not BUS_SEND_HIT:
            return
        self.write(b'$G2,HIT,1!')
        self.write(b'DONE\n')


def handle_usb(tracker, auto, bus, line, now):
    cmd = line.strip().lower()
    if not cmd:
        return
    if cmd == "off":
        auto.cancel()
        tracker.off()
        tracker.reason = "USER"
        print("OFF")
    elif cmd in ("hold", "stop"):
        auto.cancel()
        tracker.hold()
        print("HOLD")
    elif cmd == "home":
        if not tracker.enabled:
            tracker.arm()
        tracker.home()
        print("HOME")
    elif cmd in ("track", "start"):
        auto.cancel()
        tracker.bus_track()
        print("START")
    elif cmd in ("status", "s"):
        print(tracker.status_line())
    elif cmd == "info":
        print("AUTO_START", AUTO_START, "LIMITS", tracker.limits,
              "PAN_RESPONSE", tracker.pan_response, "TILT_RESPONSE", TILT_RESPONSE,
              "BUS", "UART%d@%d" % (BUS_UART_ID, BUS_BAUD),
              "SEARCH", "OFF(WAIT_IN_PLACE)")
    elif cmd == "bus":
        print("BUS uart%d@%d tgt_sent=%d cmds=%d tx=%d err=%d parked=%s last=%s" %
              (BUS_UART_ID, BUS_BAUD, bus.tgt_sent, bus.cmds, bus.tx_bytes,
               bus.stats.uart_err, bus.parked, bus.last_cmd or "-"))
    elif cmd == "tgt":
        print("TGT", tracker.report)
    elif cmd == "help":
        print("off | hold | home | track | status | info | bus | tgt | help")
    else:
        print("UNKNOWN:", cmd)


def _lock_exposure(sensor):
    """Freeze a brightness-normalised, short exposure to beat motion blur.

    Auto exposure here measured ~57 ms.  At that exposure a fast target is
    smeared across tens of pixels before the detector ever sees it -- no
    tracker can recover that.  So: let AE settle, read (exposure, gain), then
    keep the same image brightness while moving as much as possible into gain
    (the sensor tops out at ~36 dB on this OV5640).
    """
    try:
        e0 = int(sensor.get_exposure_us())
        g0 = float(sensor.get_gain_db())
    except Exception:
        e0, g0 = MAX_EXPOSURE_US, 24.0
    e1, g1 = e0, g0
    if e0 > MAX_EXPOSURE_US:
        need_db = 20.0 * math.log10(float(e0) / MAX_EXPOSURE_US)
        if g0 + need_db <= MAX_GAIN_DB:
            e1 = MAX_EXPOSURE_US
            g1 = g0 + need_db
        else:
            g1 = MAX_GAIN_DB
            e1 = int(e0 * (10.0 ** ((g0 - g1) / 20.0)))
            if e1 < 1500:
                e1 = 1500
    try:
        sensor.set_auto_exposure(False, exposure_us=int(e1))
        sensor.set_auto_gain(False, gain_db=float(g1))
        sensor.skip_frames(time=200)
    except Exception as error:
        print("EXPOSURE_LOCK_ERROR", repr(error))
    print("CAM exposure_us=%d gain_db=%.1f (auto was %d us / %.1f dB)"
          % (int(e1), g1, e0, g0))


def main():
    import sensor
    stats = Stats()
    tracker = Tracker(load_limits())
    auto = AutoStart(tracker)
    uart = None
    bus = None
    watchdog = None
    vcp = None
    if USB_CONSOLE_ENABLE:
        try:
            from pyb import USB_VCP
            vcp = USB_VCP()
        except (ImportError, AttributeError, OSError):
            vcp = None
    try:
        uart = UART(BUS_UART_ID, BUS_BAUD, read_buf_len=BUS_READ_BUF)
        uart.init(BUS_BAUD, 8, None, 1)
        bus = BusServer(uart, tracker, stats)
    except (ImportError, AttributeError, ValueError, OSError) as error:
        uart, bus = None, None
        print("BUS_INIT_ERROR", repr(error))
    try:
        sensor.reset()
        sensor.set_pixformat(sensor.RGB565)
        sensor.set_framesize(sensor.QVGA)
        sensor.set_hmirror(False)
        sensor.set_vflip(False)
        # Let auto exposure/gain/white-balance converge BEFORE locking them.
        # V10 locked them at the boot defaults, which left the image dark and
        # made every brightness threshold behave differently room-to-room.
        sensor.skip_frames(time=SETTLE_MS)
        sensor.set_auto_whitebal(False)
        _lock_exposure(sensor)
    except Exception as error:
        stats.cam_ok = 0
        stats.cam_err += 1
        print("CAMERA_INIT_ERROR", repr(error))
    if ENABLE_WATCHDOG:
        try:
            from machine import WDT
            watchdog = WDT(timeout=8000)
        except (ImportError, AttributeError, ValueError, OSError) as error:
            print("WATCHDOG_UNAVAILABLE", repr(error))
    print("V12 BULLSEYE READY; UART%d@%d P4/P5 AUTO_START %s" %
          (BUS_UART_ID, BUS_BAUD, AUTO_START))
    print("V12", "LIMITS", tracker.limits, "SEARCH", "OFF(WAIT_IN_PLACE)")
    now = time.ticks_ms()
    if AUTO_START:
        error = auto.start(now)
        if error:
            print("AUTO_START_ERROR", error)
    window = now
    last_tgt = 0
    usb_line = bytearray()
    while True:
        now = time.ticks_ms()

        # 1) USB console (OpenMV IDE terminal)
        if vcp and vcp.any():
            data = vcp.read(64)
            if data:
                for ch in data:
                    if ch in (10, 13):
                        if usb_line:
                            try:
                                handle_usb(tracker, auto, bus, usb_line.decode(), now)
                            except Exception as error:
                                print("USB_CMD_ERROR", repr(error))
                            usb_line = bytearray()
                    else:
                        usb_line.append(ch)

        # 2) Bus commands first, so replies stay inside the ESP32's 1 s window.
        #    Guarded: line noise while the ESP32 reboots must never kill the
        #    script (an uncaught error here used to take the whole board down
        #    and the watchdog then reset it ~8 s later, losing commands).
        if bus:
            try:
                bus.poll()
            except Exception as error:
                stats.uart_err += 1
                print("BUS_POLL_ERROR", repr(error))

        try:
            t_snap = time.ticks_us()
            img = sensor.snapshot()
            snap_us = time.ticks_diff(time.ticks_us(), t_snap)
            # Timestamp AFTER the capture: snapshot() can wait up to a whole
            # frame (21 ms at this rate), and every downstream consumer -- the
            # Kalman filter, the search window and the prediction -- feeds on
            # this clock.  Using the loop-start time made them up to 21 ms
            # stale, which is exactly how a fast target leaves a 60 px window.
            now = time.ticks_ms()
            target = detect(img, tracker.search_roi(img.width(), img.height(), now),
                            tracker.predicted(now), tracker.last_size)
            # Laser search window: around the target when we have one, else the
            # centre of the image (LASER_SEARCH_HALF = 0 => whole frame).
            laser_roi = None
            if LASER_SEARCH_HALF > 0:
                if tracker.last_seen is not None:
                    lcx, lcy = tracker.last_seen
                else:
                    lcx, lcy = CENTER
                half = LASER_SEARCH_HALF
                lx0 = int(max(0, lcx - half)); ly0 = int(max(0, lcy - half))
                lx1 = int(min(img.width(), lcx + half))
                ly1 = int(min(img.height(), lcy + half))
                if lx1 - lx0 >= 8 and ly1 - ly0 >= 8:
                    laser_roi = (lx0, ly0, lx1 - lx0, ly1 - ly0)
            laser = detect_laser(img, tracker.aim, laser_roi)
            tracker.update(target, laser, now)
            auto.update(now)
            stats.frame_done(now, snap_us)

            # 3) Report the vision result to the ESP32.
            if bus and time.ticks_diff(now, last_tgt) >= BUS_TGT_MIN_MS:
                last_tgt = now
                bus.send_report()
                if tracker.aligned_event:
                    bus.send_hit_and_done()
                    print("BUS: laser locked on target -> HIT + DONE")

            if target:
                img.draw_rectangle(target[2], color=(0, 255, 0), thickness=2)
                img.draw_cross(target[0], target[1], color=(255, 0, 0), size=12, thickness=2)
            if laser:
                img.draw_cross(laser[0], laser[1], color=(255, 0, 255), size=8, thickness=2)
                if target:
                    img.draw_line(laser[0], laser[1], target[0], target[1], color=(255, 0, 255))
            img.draw_cross(CENTER[0], CENTER[1], color=(255, 255, 0), size=10)
            img.draw_string(4, 4, "%s %s" % (tracker.mode, tracker.reason), color=(0, 255, 0), scale=2)
            img.draw_string(4, 24, "AIM %s ERR %d,%d" %
                            ("LASER" if tracker.laser_aim else "CENTER",
                             int(round(tracker.aim_error[0])), int(round(tracker.aim_error[1]))),
                            color=(255, 255, 0), scale=2)
            label = "TARGET %d,%d" % target[:2] if target else "NO UNIQUE TARGET"
            img.draw_string(4, 44, label, color=(0, 255, 0), scale=2)
            lx, ly = laser if laser else (-1, -1)
            img.draw_string(4, 64, "LASER %d,%d" % (lx, ly), color=(255, 0, 255), scale=2)
        except Exception as error:
            # Do NOT tear the track down on one bad frame: motion blur, a USB
            # hiccup or a draw error is transient, and tracker.off() here made
            # it look like the target "disconnected" for the rest of the run.
            stats.cam_ok = 0
            stats.cam_err += 1
            try:
                print("VISION_ERROR", repr(error))
            except Exception:
                pass
        else:
            stats.cam_ok = 1

        # 4) Bus again after the heavy frame work.
        if bus:
            try:
                bus.poll()
            except Exception as error:
                stats.uart_err += 1
                print("BUS_POLL_ERROR", repr(error))

        if time.ticks_diff(now, window) >= LOG_INTERVAL_MS:
            # A USB disconnect can make print() raise; never let telemetry kill
            # the vision loop (that looked like a random tracker disconnect).
            try:
                print("V12", tracker.mode, tracker.reason,
                      "AIM", "LASER" if tracker.laser_aim else "CENTER",
                      "TARGET", tracker.target[:2] if tracker.target else None,
                      "V", (int(round(tracker.velocity[0])),
                            int(round(tracker.velocity[1]))),
                      "FPS", stats.fps_x10 / 10.0,
                      "LV", _LAST_LEVEL[0],
                      "LASER", tracker.laser,
                      "TGT", bus.last_cmd if bus else "-")
                # Per-second rejection breakdown: this is how we tell "the
                # gimbal was too slow" apart from "the detector threw the
                # target away".
                print("REJ size=%d span=%d contrast=%d glyph=%d dark=%d ring=%d jump=%d acc=%d"
                      % (REJECT["size"], REJECT["span"], REJECT["contrast"],
                         REJECT["glyph"], REJECT["dark"], REJECT["ring"],
                         REJECT["jump"], REJECT["accept"]))
            except Exception:
                pass
            for _k in REJECT:
                REJECT[_k] = 0
            window = now
        if watchdog:
            watchdog.feed()


if __name__ == "__main__":
    main()
