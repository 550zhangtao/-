"""V6 FAST: OpenMV vision front-end for the KPZ-ESP32 car over UART3.

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
TRACK_ROI_HALF = 150     # px half-size of the window handed to find_blobs
ROI_LOST_FRAMES = 8      # ROI misses before falling back to a full-frame search
# When re-acquiring after a loss, how much an apparent disk-size mismatch counts
# against a candidate, in "pixels of distance".  Raising it trusts the
# remembered target size more than proximity to the last known position.
ROI_SIZE_WEIGHT = 2.0

# ---- target motion filter (alpha-beta) -------------------------------------
# Velocity is what makes a FAST target trackable: it feeds the feed-forward term
# on the ESP32 and lets us predict where the target will be by the time a servo
# command actually takes effect.  Without it the loop can only chase the last
# measured position and always arrives late.
AB_ALPHA = 0.55          # position correction gain
AB_BETA = 0.18           # velocity correction gain
AB_MAX_SPEED = 4000.0    # px/s; reject absurd velocity estimates (bad frames)
AIM_LEAD_MS = 60.0       # how far ahead to look when centring the search window

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
    """Return the target (dot cx, dot cy, disk rect) or None.

    Without `prefer` (nothing locked yet) the original strict rule applies --
    exactly one candidate -- so a cold start cannot grab a random object.

    With `prefer` (we had a lock before) candidates are ranked by distance to
    the last known position plus how similar their apparent disk size is, and
    the best one wins *even when the frame contains other white/orange objects*.

    That second part matters: the earlier version gave up whenever more than one
    candidate was present, so once the target was lost in a scene that also
    contains decoys (a plate with fruit, an orange box) it could never
    re-acquire -- the loop stayed at run=0 forever and looked "stuck".
    """
    # MicroPython's find_blobs() rejects roi=None ("object 'NoneType' isn't a
    # tuple or list"), so an absent window becomes an explicit full-frame rect.
    blob_roi = search_roi
    if blob_roi is None:
        blob_roi = (0, 0, img.width(), img.height())
    rx, ry, rw, rh = blob_roi
    found = []
    for disk in img.find_blobs([WHITE], roi=blob_roi, pixels_threshold=250,
                               area_threshold=400, merge=False):
        w, h = disk.w(), disk.h()
        if not (20 <= w <= 220 and 20 <= h <= 220):
            continue
        # The disk must be complete.  When searching a window, "complete" means
        # inside the WINDOW, not just the image: a disk cut by the ROI border
        # would report a wrong bounding box (and therefore a wrong centre).
        if (disk.x() <= rx + 2 or disk.y() <= ry + 2
                or disk.x() + w >= rx + rw - 2 or disk.y() + h >= ry + rh - 2):
            continue
        if not 0.65 <= w / h <= 1.45 or not 0.55 <= disk.density() <= 0.92:
            continue
        dot_roi = (disk.x() + w // 4, disk.y() + h // 4, w // 2, h // 2)
        for dot in img.find_blobs([ORANGE], roi=dot_roi, pixels_threshold=5, area_threshold=6, merge=False):
            if not 0.45 <= dot.w() / dot.h() <= 1.8 or dot.density() < 0.4:
                continue
            if not 0.002 <= dot.pixels() / disk.pixels() <= 0.08:
                continue
            if abs(dot.cx() - disk.cx()) > w * 0.18 or abs(dot.cy() - disk.cy()) > h * 0.18:
                continue
            found.append((dot.cx(), dot.cy(), disk.rect()))
    if not found:
        return None
    if prefer is None:
        return found[0] if len(found) == 1 else None
    if prefer_size is None:
        found.sort(key=lambda f: (f[0] - prefer[0]) ** 2 + (f[1] - prefer[1]) ** 2)
    else:
        found.sort(key=lambda f: (((f[0] - prefer[0]) ** 2 + (f[1] - prefer[1]) ** 2) ** 0.5
                                  + ROI_SIZE_WEIGHT * abs(f[2][2] - prefer_size)))
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
        self.filtered = None
        self.velocity = [0.0, 0.0]    # px/s, from the alpha-beta tracker
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

    def search_roi(self, width, height):
        """Window handed to find_blobs, centred on where the target is *now*.

        Using the predicted position (not the last measured one) keeps a fast
        target inside the window, and a small window is ~2x faster than a
        full-frame scan (measured: 39 ms vs 18 ms on VGA).
        """
        if self.last_seen is None or self.miss >= ROI_LOST_FRAMES:
            return None
        lead = AIM_LEAD_MS / 1000.0
        cx = self.last_seen[0] + self.velocity[0] * lead
        cy = self.last_seen[1] + self.velocity[1] * lead
        half = TRACK_ROI_HALF
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
    def build_report(self):
        """$G2,TGT,<cx>,<cy>,<vx>,<vy>,<lx>,<ly>,<aim>,<run>!

        vx/vy are the filtered target velocity in px/s.  The ESP32 uses them for
        the feed-forward term and for look-ahead, which is what lets a small
        servo gimbal stay on a fast target instead of always arriving late.
        """
        cx, cy = (int(round(self.filtered[0])), int(round(self.filtered[1]))) \
            if self.filtered else (-1, -1)
        vx, vy = (int(round(self.velocity[0])), int(round(self.velocity[1]))) \
            if self.filtered else (0, 0)
        lx, ly = (int(self.laser[0]), int(self.laser[1])) if self.laser else (-1, -1)
        aim = 1 if (self.laser_aim and self.aim) else 0
        run = 1 if self.mode == "TRACK" else 0
        return ("$G2,TGT,%d,%d,%d,%d,%d,%d,%d,%d!"
                % (cx, cy, vx, vy, lx, ly, aim, run)).encode()

    # ---------------------------------------------------------- main step
    def update(self, target, laser, now):
        dt = 0 if self.last_time is None else min(0.1, max(0, time.ticks_diff(now, self.last_time) / 1000))
        self.last_time = now
        previous = self.target
        self.target = target
        self.laser = laser
        if target:
            self.last_seen = (target[0], target[1])
            self.last_size = target[2][2]
            self.miss = 0
        else:
            self.miss += 1
        if target:
            # Keep a visible moving disk; reject discontinuous jumps/size changes.
            gate = max(40, min(160, 2 * max(target[2][2], target[2][3])))
            continuous = (previous and abs(target[0] - previous[0]) <= gate
                          and abs(target[1] - previous[1]) <= gate
                          and 0.5 <= target[2][2] / previous[2][2] <= 2
                          and 0.5 <= target[2][3] / previous[2][3] <= 2)
            self.stable = min(100, self.stable + 1) if continuous else 1
            if not continuous or self.filtered is None:
                # (re)seed the filter; keep any velocity we already trust
                self.filtered = [float(target[0]), float(target[1])]
                if not continuous:
                    self.velocity = [0.0, 0.0]
            else:
                # alpha-beta tracker: smooths position AND estimates velocity,
                # which the ESP32 uses for feed-forward + look-ahead.  This is
                # what lets the loop lead a fast target instead of chasing it.
                step = dt if dt > 0.001 else 0.001
                for i in (0, 1):
                    predicted = self.filtered[i] + self.velocity[i] * step
                    residual = target[i] - predicted
                    self.filtered[i] = predicted + AB_ALPHA * residual
                    self.velocity[i] += (AB_BETA / step) * residual
                    if self.velocity[i] > AB_MAX_SPEED:
                        self.velocity[i] = AB_MAX_SPEED
                    elif self.velocity[i] < -AB_MAX_SPEED:
                        self.velocity[i] = -AB_MAX_SPEED
        else:
            self.stable, self.filtered = 0, None
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
    print("V6 FAST READY; UART%d@%d P4/P5 AUTO_START %s" %
          (BUS_UART_ID, BUS_BAUD, AUTO_START))
    print("V6", "LIMITS", tracker.limits, "SEARCH", "OFF(WAIT_IN_PLACE)")
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
            target = detect(img, tracker.search_roi(img.width(), img.height()),
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
            print("V6", tracker.mode, tracker.reason,
                  "AIM", "LASER" if tracker.laser_aim else "CENTER",
                  "TARGET", tracker.target[:2] if tracker.target else None,
                  "V", (int(round(tracker.velocity[0])), int(round(tracker.velocity[1]))),
                  "FPS", stats.fps_x10 / 10.0,
                  "LASER", tracker.laser, "TGT", bus.last_cmd if bus else "-")
            window = now
        if watchdog:
            watchdog.feed()
        time.sleep_ms(2)


if __name__ == "__main__":
    main()
