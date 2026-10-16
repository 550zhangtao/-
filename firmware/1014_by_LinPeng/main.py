"""V9 ROBUST: OpenMV vision front-end for the KPZ-ESP32 car over UART3.

Deployment: open this file in OpenMV IDE and SAVE TO BOARD AS main.py.

WHAT THIS VERSION CHANGES (vs main(5).py = V4 AUTO + INIT-ALIGN)
---------------------------------------------------------------
main(5).py was deliberately standalone: it drove the pan/tilt servos itself on
P7/P8 and had no bus at all ("No ESP32, no UART bus ... required").  In this
version the ESP32 owns the gimbals (GPIO27/14) and this script only *looks*.
This is the agreed split: OpenMV reports target coordinates, the ESP32 runs the
aiming PID and drives the servos.

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

  * It no longer touches P7/P8 at all.  Power-on auto-track is OFF
    (AUTO_START = False) so the ESP32 decides when tracking starts via
    $G2,TRACK; set AUTO_START = True to keep the old stand-alone behaviour.

The two-phase aiming *decision* (centre the target first, then switch the aim
reference to the laser spot) is unchanged and still lives here, because it is a
vision decision; only the motion control moved to the ESP32.  That is why the
`aim` field exists: it tells the ESP32 which reference to servo towards.

Direction/response signs and the laser threshold are unchanged from main(5).py.
"""
import time
from pyb import UART

# ---------------------------------------------------------------- behaviour
AUTO_START = True    # Track on power-up without waiting for $G2,TRACK.
# Why True: if anything ever stalls this script the hardware watchdog resets the
# board and main.py starts again.  With AUTO_START it comes back already
# tracking (run=1), so the ESP32 -- whose mode gates whether the servos move --
# resumes immediately.  With False it would come back in OFF, waiting for a
# $G2,TRACK that may never be re-sent, and the gimbal would sit still forever.
# Set it back to False if you want the OpenMV to stay passive until commanded.
RUN_LIMITS = ((740, 2260), (1200, 1800))  # Recorded confirmed assembly travel, us.
BOOT_PAN_RESPONSE = 1  # Target-image dx sign for increasing pan PWM.
TILT_RESPONSE = 1      # Target-image dy sign for increasing tilt PWM.
SKIP_FRAMES_MS = 300
HOME_SETTLE_MS = 300
AUTO_STABLE_FRAMES = 3
TRACK_STABLE_FRAMES = 3
LOG_INTERVAL_MS = 1000
ENABLE_WATCHDOG = True
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

# Once locked, only look near the previous position.  The scene contains other
# white objects with orange/yellow parts (a plate with fruit, an orange box,
# white walls/roll/shirt) and without this gate the detector happily switches to
# one of them the moment the real target leaves the frame, which makes the aim
# loop chase a phantom (observed as the reported target teleporting).
# When re-acquiring after a loss, how much an apparent disk-size mismatch counts
# against a candidate, in "pixels of distance".  Raising it trusts the
# remembered target size more than proximity to the last known position.
ROI_SIZE_WEIGHT = 2.0

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
KF_ACCEL_STD = 900.0     # px/s^2: manoeuvre level the target may have
KF_MEAS_STD = 3.0        # px: detection noise on a good frame
KF_MAX_SPEED = 6000.0    # px/s sanity clamp
KF_MIN_DT = 0.001        # s

# ---- search window: predictive and adaptive --------------------------------
# Measured on this rig: a +-150px window costs ~4ms per frame, a FULL VGA scan
# ~42ms.  Going full-frame therefore halves the loop rate (46.6 -> ~21fps)
# exactly during the moments a fast target is hardest to hold.  So while a track
# is alive we never go full-frame: the window simply grows with the uncertainty.
ROI_BASE_HALF = 70       # px: minimum half-window
ROI_SIGMA_K = 5.0        # half-window += this * position sigma (px)
ROI_MAX_HALF = 250       # px: cap, keeps the scan well inside one frame period
ROI_DEAD_MS = 700.0      # ms with no sighting before we give up and re-scan
ROI_LEAD_MS = 60.0       # ms of lead used when placing the window

# ---- 锁定后的宽容判定 (V9) --------------------------------------------------
# 实测定位: "速度稍快就丢靶" 的主因不是舵机, 而是检出条件太苛刻 —— 冷启动那一套
# 要求 8 个几何条件在同一帧同时成立. 运动模糊会把圆盘拉扁(长宽比失败)、降低密度
# (密度失败)、糊掉小橙点(找不到橙点); 而靶一跑快、预测稍有偏差, 圆盘就被搜索窗
# 裁切, 于是被整个丢弃 —— 哪怕它在画面里清清楚楚. 任何一条不满足就是"整帧没有
# 测量", 跟踪随即断掉.
# 所以: 冷启动仍然严格(避免抓到别的白物体), 一旦锁定就放宽几何门限、橙点改为
# 加分项、并允许被裁切的圆盘(用记忆半径反推圆心).
DOT_SCORE_BONUS = 45.0        # "px 距离"当量: 有橙点的候选优先
# 丢失后保留"上次位置"作为弱参考的时长. 冷启动那套"全画面唯一"的规则在杂物环境里
# 几乎不可能满足, 于是"短暂丢失"就变成"永久丢失". 在这段时间内改为按"离丢失位置
# 的距离 + 尺寸相似度 + 橙点加分"排序取最优, 才能真正重新捕获.
REACQ_MEMORY_MS = 3000.0
# 重捕窗口半径. 全画面扫描实测 ~42ms(20fps), 而有界窗口 ~26ms(38fps) —— 丢失重捕
# 正是最需要帧率的时候, 所以这段时间用有界窗口围着"上次位置"找, 而不是扫全画面.
REACQ_ROI_HALF = 220
RELAX_ASPECT_MIN = 0.42       # 锁定后允许被模糊拉扁/拉长
RELAX_ASPECT_MAX = 2.20
RELAX_DENSITY_MIN = 0.32      # 模糊会让密度掉下来
RELAX_DENSITY_MAX = 0.97

# 每帧被拒原因计数 —— 用来量化"到底为什么丢", 每秒随日志打印一次
REJECT = {"size": 0, "clip": 0, "shape": 0, "dot": 0, "span": 0, "accept": 0}

# ---- coasting --------------------------------------------------------------
# The single biggest cause of "lost it at speed": one missed detection made the
# vision side report run=0 and the ESP32 freeze instantly, so the gimbal stopped
# dead and the target ran away.  Instead keep the track alive briefly, report the
# PREDICTED spot with run=1, and let the ESP32 keep moving.
TGT_COAST_MS = 260.0     # ms of prediction-only output after the last sighting

# --------------------------------------------------------------- vision
WHITE = (48, 100, -24, 12, -24, 30)
ORANGE = (12, 90, 12, 65, 5, 70)  # A <= 65: orange can never pass the laser gate.
# Strict laser: bright + VERY red + not yellowish. Orange (A<=65), dark red
# (L<70) and yellow-red (B>55) are all rejected. Retune on site if needed.
LASER = (70, 100, 66, 100, 5, 55)
LASER_MIN_PX = 8    # Tiny noise blobs do not count.
LASER_MAX_PX = 900
LASER_MAX_SIDE = 40
LASER_JUMP_PX = 100  # Spot is rigid with the camera; larger steps are noise.
AIM_ALPHA = 0.5
# The laser search is the second full-frame blob scan of every frame and costs
# about as much as the target search.  Restrict it to a window around the
# target: when the laser is doing its job it is ON the target, and this roughly
# doubles the achievable frame rate.  Set to 0 to scan the whole frame.
LASER_SEARCH_HALF = 110
ALIGNED_PX = 12
LIMITS = ((1460, 1540), (1480, 1550))
PROBE_LIMITS = ((500, 2500), (1100, 1900))
CENTER = (320, 240)
# Measured on this gimbal: the pan servo ignores commands below ~40-60 us and
# then jumps ~120-180 px, so a 12 px deadband asks for moves the hardware cannot
# make and the loop hunts forever.  Keep the deadband at the mechanical limit;
# lower it only if the linkage/servo resolution is improved.
DEADBAND_PX = 120
RESTART_PX = 150
GAIN_US_PER_SEC_PER_PX = 1.8
MAX_US_PER_SEC = 800      # Pan speed cap, us/s.
TILT_MAX_US_PER_SEC = 200  # Pitch speed cap, us/s.
TILT_GAIN_CAP = 1.2
ACCEL_US_PER_SEC2 = 2400


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


def detect(img, search_roi=None, prefer=None, prefer_size=None):
    """Return (cx, cy, disk_rect, dot_bonus) or None.

    Two modes, and the difference is the whole point:

    * COLD (prefer is None) -- the original strict gate: a complete, round,
      dense white disk carrying a small centred orange dot, and it must be the
      ONLY candidate.  A cold start therefore cannot grab a random object.
    * LOCKED (prefer given) -- identity comes from position and size, not from
      the orange dot.  Geometry gates are widened and a disk clipped by the
      frame OR by the search window is still accepted, with its centre
      reconstructed from the remembered radius.

    Why LOCKED mode exists: measured on this rig, a target moving only
    moderately fast was lost constantly, and the cause was not the servos -- it
    was that eight geometric conditions had to hold in the same frame.  Motion
    blur breaks the aspect and density gates, blur swallows the small orange
    dot, and a target that outruns the prediction by a few pixels has its
    bounding box clipped by the search window and used to be thrown away even
    though it was plainly visible.  Each of those turned into "no measurement
    this frame" and the track died.
    """
    # MicroPython's find_blobs() rejects roi=None, so an absent window becomes an
    # explicit full-frame rect.
    blob_roi = search_roi
    if blob_roi is None:
        blob_roi = (0, 0, img.width(), img.height())
    rx, ry, rw, rh = blob_roi
    locked = prefer is not None
    radius = 0.0
    if locked and prefer_size:
        radius = prefer_size / 2.0
    REJECT["clip"] = REJECT["clip"]      # (kept for symmetry; counters below)
    found = []

    for disk in img.find_blobs([WHITE], roi=blob_roi, pixels_threshold=250,
                               area_threshold=400, merge=False):
        w, h = disk.w(), disk.h()
        x, y = disk.x(), disk.y()

        # "clipped" = cut by the search window (or the frame edge when the window
        # is the whole frame).  A clipped disk reports a truncated bounding box,
        # so its centre cannot be read directly -- but it can be reconstructed.
        left = x <= rx + 2
        right = x + w >= rx + rw - 2
        top = y <= ry + 2
        bottom = y + h >= ry + rh - 2
        clipped = left or right or top or bottom

        # Size: judge by the LONGER side, because clipping truncates one of them.
        span = w if w > h else h
        if not (20 <= span <= 220):
            REJECT["size"] += 1
            continue

        if clipped:
            REJECT["clip"] += 1
            if not locked or radius <= 0.0:
                continue          # cold start: a partial disk proves nothing

        if not clipped:
            if locked:
                if not (RELAX_ASPECT_MIN <= w / h <= RELAX_ASPECT_MAX):
                    REJECT["shape"] += 1
                    continue
                if not (RELAX_DENSITY_MIN <= disk.density() <= RELAX_DENSITY_MAX):
                    REJECT["shape"] += 1
                    continue
            else:
                if not (0.65 <= w / h <= 1.45):
                    REJECT["shape"] += 1
                    continue
                if not (0.55 <= disk.density() <= 0.92):
                    REJECT["shape"] += 1
                    continue

        # Centre: direct when the disk is whole, reconstructed when it is cut.
        # NOTE: the reconstruction must stay INTEGER.  MicroPython's draw_*()
        # reject floats, and a float centre here made every frame raise
        # "can't convert float to int" -- which silently killed the whole vision
        # loop and looked exactly like "the tracker cannot see the target".
        if clipped:
            if left and not right:
                cx = int(x + w - radius + 0.5)
            elif right and not left:
                cx = int(x + radius + 0.5)
            else:
                cx = x + w // 2
            if top and not bottom:
                cy = int(y + h - radius + 0.5)
            elif bottom and not top:
                cy = int(y + radius + 0.5)
            else:
                cy = y + h // 2
        else:
            cx, cy = disk.cx(), disk.cy()

        if not (0 <= cx < img.width() and 0 <= cy < img.height()):
            REJECT["span"] += 1
            continue

        # Orange dot: the identity proof when cold, a score bonus when locked.
        has_dot = False
        dw = w // 2
        dh = h // 2
        if dw >= 2 and dh >= 2:
            dot_roi = (x + w // 4, y + h // 4, dw, dh)
            for dot in img.find_blobs([ORANGE], roi=dot_roi, pixels_threshold=5,
                                      area_threshold=6, merge=False):
                if not 0.45 <= dot.w() / dot.h() <= 1.8 or dot.density() < 0.4:
                    continue
                if not 0.002 <= dot.pixels() / disk.pixels() <= 0.08:
                    continue
                if abs(dot.cx() - cx) > w * 0.18 or abs(dot.cy() - cy) > h * 0.18:
                    continue
                has_dot = True
                break
        if not has_dot and not locked:
            REJECT["dot"] += 1
            continue

        REJECT["accept"] += 1
        found.append((cx, cy, disk.rect(), DOT_SCORE_BONUS if has_dot else 0.0))

    if not found:
        return None
    if prefer is None:
        # Cold acquisition: refuse to guess between several candidates.
        return found[0] if len(found) == 1 else None
    if prefer_size is None:
        found.sort(key=lambda f: (f[0] - prefer[0]) ** 2 + (f[1] - prefer[1]) ** 2 - f[3])
    else:
        found.sort(key=lambda f: (((f[0] - prefer[0]) ** 2 + (f[1] - prefer[1]) ** 2) ** 0.5
                                  + ROI_SIZE_WEIGHT * abs(f[2][2] - prefer_size) - f[3]))
    return found[0]


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
        if spot.pixels() > LASER_MAX_PX or spot.density() < 0.60:
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
        half += int(max(abs(self.kf.v[0]), abs(self.kf.v[1])) * (age / 1000.0))
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
        """
        if now is None:
            now = self._clock()
        cx = cy = -1
        vx = vy = 0
        run = 0
        if self.moving_allowed(now):
            spot = self.kf.pos_at(now)
            if spot is not None:
                cx = int(round(spot[0])); cy = int(round(spot[1]))
                vx = int(round(self.kf.v[0])); vy = int(round(self.kf.v[1]))
                run = 1
        lx, ly = (int(self.laser[0]), int(self.laser[1])) if self.laser else (-1, -1)
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
            gate = max(40, min(160, 2 * max(target[2][2], target[2][3])))
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


def main():
    import sensor
    stats = Stats()
    tracker = Tracker(load_limits())
    auto = AutoStart(tracker)
    uart = None
    bus = None
    watchdog = None
    vcp = None
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
        sensor.set_framesize(sensor.VGA)
        sensor.set_hmirror(False)
        sensor.set_vflip(False)
        sensor.set_auto_gain(False)
        sensor.set_auto_whitebal(False)
        sensor.skip_frames(time=SKIP_FRAMES_MS)
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
    print("V9 ROBUST READY; UART%d@%d P4/P5 AUTO_START %s" %
          (BUS_UART_ID, BUS_BAUD, AUTO_START))
    print("V9", "LIMITS", tracker.limits, "SEARCH", "OFF(WAIT_IN_PLACE)")
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
            target = detect(img, tracker.search_roi(img.width(), img.height(), now),
                            tracker.last_seen, tracker.last_size)
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
            img.draw_cross(320, 240, color=(255, 255, 0), size=10)
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
            stats.cam_ok = 0
            stats.cam_err += 1
            auto.cancel()
            tracker.off()
            tracker.reason = "CAMERA_OR_CONTROL_ERROR"
            print("VISION_ERROR", repr(error))
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
            print("V9", tracker.mode, tracker.reason,
                  "AIM", "LASER" if tracker.laser_aim else "CENTER",
                  "TARGET", tracker.target[:2] if tracker.target else None,
                  "V", (int(round(tracker.velocity[0])), int(round(tracker.velocity[1]))),
                  "FPS", stats.fps_x10 / 10.0,
                  "LASER", tracker.laser, "TGT", bus.last_cmd if bus else "-")
            # Per-second rejection breakdown: this is how we tell "the gimbal
            # was too slow" apart from "the detector threw the target away".
            print("REJ size=%d clip=%d shape=%d dot=%d acc=%d"
                  % (REJECT["size"], REJECT["clip"], REJECT["shape"],
                     REJECT["dot"], REJECT["accept"]))
            for _k in REJECT:
                REJECT[_k] = 0
            window = now
        if watchdog:
            watchdog.feed()
        time.sleep_ms(2)


if __name__ == "__main__":
    main()
    