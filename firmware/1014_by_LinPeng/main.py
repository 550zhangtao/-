"""V4 AUTO + INIT-ALIGN: OpenMV-only laser-aim pan/tilt tracker.

Deployment: open this file in OpenMV IDE and SAVE TO BOARD AS main.py.

Runs ENTIRELY on the OpenMV board:
  - camera capture (VGA 640x480 RGB565),
  - white-disk + orange-center target (靶心) detection,
  - red-laser-spot detection with a STRICT threshold: only a bright, VERY
    red, compact spot passes. Ordinary orange objects (and the orange
    target center, A <= 65) cannot pass the laser gate (A >= 66),
  - P7 (pan) / P8 (tilt) servo control,
  - POWER-ON AUTO-START: camera warmup -> home -> settle -> first valid
    target frames -> track. No host command is required.

Aiming is done in TWO PHASES (the laser is assumed to be always ON):
  1. INITIAL ALIGNMENT (center mode): before the laser has been locked,
     the aim reference is the IMAGE CENTER. The gimbal first moves the
     target to the screen center and keeps it centered there.
  2. LASER AIM (laser mode): once the target is centered (both axes within
     the deadband for ALIGN_LOCK_FRAMES consecutive frames) AND a laser
     spot is seen, the reference switches to the FILTERED LASER SPOT. From
     that moment on the gimbal keeps the laser ON the target wherever it
     moves; it no longer tries to center the target on screen.
  - Laser lost (no spot in view) -> back to center mode immediately: the
    target is kept at the screen center again.
  - Target lost -> WAIT in place, no scanning; tracking resumes the moment
    the target reappears.

No ESP32, no UART bus, no host PC, no config file, no SD card are required.

Optional manual control via the OpenMV IDE serial terminal (USB VCP),
one command per line + Enter:
    off      power off both servos (PWM off) and stop everything
    hold     hold current position (cancel tracking)
    home     power on and move to 1500/1500, then hold
    track    (re)start auto acquire -> aim (arms if needed)
    status   print current mode/reason/PWM/target/laser/error
    info     print configuration summary
    help     print this list

Direction is configured by constants, not runtime calibration:
  BOOT_PAN_RESPONSE = 1  target-image dx sign for increasing pan PWM,
  TILT_RESPONSE    = 1  target-image dy sign for increasing tilt PWM.
If you remount the servos or mirror/flip the image, edit these two.

LASER threshold is tuned STRICT: L 70-100, A 66-100, B 5-55. If the real
laser is not detected at all, lower A_MIN / L_MIN (or raise B_MAX) with
OpenMV IDE Tools->Machine Vision->Threshold Editor until only the laser
passes. Spots that jump more than LASER_JUMP_PX from the previous
reference are treated as reflections; ambiguous multiple spots are
ignored.

Derived from main(4).py (V3 WAIT-MODE):
  - AUTO_START back to True (power-on auto-home + auto-track),
  - added the two-phase aim: center the target first (initial alignment),
    then switch to laser aim once the target is centered and a laser spot
    is seen; the laser then stays on the target wherever it moves,
  - laser lost -> back to center mode (target centered on screen) at once,
  - target lost -> WAIT in place, no scanning.
"""
import time
from pyb import Pin, Servo

AUTO_START = True   # Power-on: auto home, auto settle, auto track.
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
ALIGNED_PX = 12
LIMITS = ((1460, 1540), (1480, 1550))
PROBE_LIMITS = ((500, 2500), (1100, 1900))
CENTER = (320, 240)
DEADBAND_PX = 12
RESTART_PX = 18
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


def detect(img):
    """Return the single target (dot cx, dot cy, disk rect) or None."""
    found = []
    for disk in img.find_blobs([WHITE], pixels_threshold=250, area_threshold=400, merge=False):
        w, h = disk.w(), disk.h()
        if not (20 <= w <= 220 and 20 <= h <= 220):
            continue
        if disk.x() <= 2 or disk.y() <= 2 or disk.x() + w >= img.width() - 2 or disk.y() + h >= img.height() - 2:
            continue  # A clipped disk is not a complete geometric match.
        if not 0.65 <= w / h <= 1.45 or not 0.55 <= disk.density() <= 0.92:
            continue
        roi = (disk.x() + w // 4, disk.y() + h // 4, w // 2, h // 2)
        for dot in img.find_blobs([ORANGE], roi=roi, pixels_threshold=5, area_threshold=6, merge=False):
            if not 0.45 <= dot.w() / dot.h() <= 1.8 or dot.density() < 0.4:
                continue
            if not 0.002 <= dot.pixels() / disk.pixels() <= 0.08:
                continue
            if abs(dot.cx() - disk.cx()) > w * 0.18 or abs(dot.cy() - disk.cy()) > h * 0.18:
                continue
            found.append((dot.cx(), dot.cy(), disk.rect()))
    return found[0] if len(found) == 1 else None


def detect_laser(img, previous):
    """Return the single valid laser spot (x, y) or None.

    Only a bright, VERY red, compact spot passes (LASER threshold), so
    orange objects and the orange target center cannot steal the aim
    reference. No target-center exclusion is applied here: when the laser
    is already ON the target its spot still counts, which keeps the gimbal
    stable and reports ALIGNED.
    """
    spots = []
    for spot in img.find_blobs([LASER], pixels_threshold=LASER_MIN_PX,
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
    def __init__(self, limits=None):
        self.limits = [list(pair) for pair in (limits or LIMITS)]
        self.servos = None
        self.position = [1500.0, 1500.0]
        self.sent = [1500, 1500]
        self.pan_response = BOOT_PAN_RESPONSE
        self.mode = "OFF"
        self.reason = "BOOT"
        self.centered = [False, False]
        self.velocity = [0.0, 0.0]
        self.last_time = None
        self.target = None
        self.laser = None
        self.aim = None  # Filtered laser spot; None means "center the target".
        self.aim_error = (0.0, 0.0)
        self.stable = 0
        self.filtered = None
        self.at_limit = False
        self.laser_aim = False   # True: aim at the laser spot (laser mode).
        self.align_frames = 0    # Consecutive centered frames before laser lock.
        self.off()

    def off(self):
        Pin("P7", Pin.OUT_PP, value=0)
        Pin("P8", Pin.OUT_PP, value=0)
        self.servos = None
        self.mode = "OFF"
        self.centered = [False, False]
        self.velocity = [0.0, 0.0]
        self.laser_aim = False
        self.align_frames = 0

    def arm(self):
        if self.servos:
            return False
        self.servos = [Servo(1), Servo(2)]
        for servo in self.servos:
            servo.calibration(500, 2500, 1500)
            servo.pulse_width(1500)
        self.position = [1500.0, 1500.0]
        self.sent = [1500, 1500]
        self.centered = [False, False]
        self.mode = "HOLD"
        self.reason = "HOME"
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
        self.velocity = [0.0, 0.0]
        self.position = [float(value) for value in self.sent]
        self.laser_aim = False
        self.align_frames = 0
        return True

    def move(self, axis, value, envelope=None):
        low, high = (envelope or self.limits)[axis]
        value = max(low, min(high, value))
        self.position[axis] = float(value)
        quantized = max(low, min(high, int(round(value / 10)) * 10))
        if quantized != self.sent[axis]:
            self.servos[axis].pulse_width(quantized)
            self.sent[axis] = quantized

    def status_line(self):
        pan, tilt = self.sent if self.servos else (0, 0)
        x, y = self.target[:2] if self.target else (-1, -1)
        lx, ly = self.laser if self.laser else (-1, -1)
        return ("MODE %s REASON %s AIM %s PWM %d,%d TARGET %d,%d LASER %d,%d ERR %d,%d" %
                (self.mode, self.reason, "LASER" if self.laser_aim else "CENTER",
                 pan, tilt, x, y, lx, ly,
                 int(round(self.aim_error[0])), int(round(self.aim_error[1]))))

    def update(self, target, laser, now):
        dt = 0 if self.last_time is None else min(0.1, max(0, time.ticks_diff(now, self.last_time) / 1000))
        self.last_time = now
        previous = self.target
        self.target = target
        self.laser = laser
        if target:
            # Keep a visible moving disk; reject discontinuous jumps/size changes.
            gate = max(40, min(160, 2 * max(target[2][2], target[2][3])))
            continuous = (previous and abs(target[0] - previous[0]) <= gate
                          and abs(target[1] - previous[1]) <= gate
                          and 0.5 <= target[2][2] / previous[2][2] <= 2
                          and 0.5 <= target[2][3] / previous[2][3] <= 2)
            self.stable = min(100, self.stable + 1) if continuous else 1
            if not continuous or self.filtered is None:
                self.filtered = [float(target[0]), float(target[1])]
            else:
                for i in (0, 1):
                    alpha = 0.65 if abs(target[i] - CENTER[i]) > RESTART_PX else 0.35
                    self.filtered[i] += alpha * (target[i] - self.filtered[i])
        else:
            self.stable, self.filtered = 0, None
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
        if self.mode not in ("TRACK", "WAIT"):
            return
        if not target:
            # Target lost: WAIT in place, no scanning. Tracking resumes
            # automatically the moment a valid target reappears.
            self.mode = "WAIT"
            self.reason = "WAIT_TARGET"
            self.velocity = [0.0, 0.0]
            self.centered = [False, False]
            self.position = [float(value) for value in self.sent]
            return
        if self.stable < 2:
            self.mode = "WAIT"
            self.reason = "CONFIRM_TARGET"
            self.position = [float(value) for value in self.sent]
            self.velocity = [0.0, 0.0]
            self.centered = [False, False]
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
        self.mode, self.reason = "TRACK", "FOLLOW"
        self.at_limit = False
        for axis, response in ((0, self.pan_response), (1, TILT_RESPONSE)):
            error = self.filtered[axis] - ref[axis]
            threshold = RESTART_PX if self.centered[axis] else DEADBAND_PX
            if abs(error) <= threshold:
                self.centered[axis] = True
                self.position[axis] = float(self.sent[axis])
                self.velocity[axis] = 0.0
                continue
            self.centered[axis] = False
            gain = GAIN_US_PER_SEC_PER_PX if abs(error) <= 64 else (2.8 if abs(error) <= 120 else 4.0)
            if axis == 1:
                gain = min(gain, TILT_GAIN_CAP)
            cap = MAX_US_PER_SEC if axis == 0 else TILT_MAX_US_PER_SEC
            desired = max(-cap, min(cap, -response * gain * error))
            old = self.velocity[axis]
            if old * desired <= 0 or abs(desired) < abs(old):
                old = 0.0 if old * desired <= 0 else desired
            velocity = max(old - ACCEL_US_PER_SEC2 * dt, min(old + ACCEL_US_PER_SEC2 * dt, desired))
            self.velocity[axis] = velocity
            target_us = self.position[axis] + velocity * dt
            if not self.limits[axis][0] <= target_us <= self.limits[axis][1]:
                self.at_limit = True
            self.move(axis, target_us)
        self.aim_error = (self.filtered[0] - ref[0], self.filtered[1] - ref[1])
        if self.at_limit:
            self.reason = "LIMIT"
        elif self.laser_aim:
            self.reason = "ALIGNED" if max(abs(self.aim_error[0]), abs(self.aim_error[1])) <= ALIGNED_PX else "LASER_AIM"
        else:
            self.reason = "CENTERED" if max(abs(self.aim_error[0]), abs(self.aim_error[1])) <= ALIGNED_PX else "CENTER_TARGET"


class AutoStart:
    """Power-on auto start (also used by the 'track' USB command)."""
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
        if not t.servos and not t.arm():
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
        # No target: stay in HOLD and wait. No scanning is ever performed.


def handle_usb(tracker, auto, line, now):
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
        if tracker.servos:
            tracker.velocity = [0.0, 0.0]
            tracker.mode = "HOLD"
            tracker.reason = "USER"
            print("HOLD")
        else:
            print("NOT_ARMED")
    elif cmd == "home":
        if not tracker.servos:
            tracker.arm()
            print("HOME")
        else:
            print("ALREADY_ARMED; use off first to re-home")
    elif cmd in ("track", "start"):
        auto.cancel()
        error = auto.start(now)
        print("START" if error is None else "START_ERR " + error)
    elif cmd in ("status", "s"):
        print(tracker.status_line())
    elif cmd == "info":
        print("AUTO_START", AUTO_START, "LIMITS", tracker.limits,
              "PAN_RESPONSE", tracker.pan_response, "TILT_RESPONSE", TILT_RESPONSE,
              "SEARCH", "OFF(WAIT_IN_PLACE)")
    elif cmd == "help":
        print("off | hold | home | track | status | info | help")
    else:
        print("UNKNOWN:", cmd)


def main():
    import sensor
    tracker = Tracker(load_limits())
    auto = AutoStart(tracker)
    watchdog = None
    vcp = None
    try:
        from pyb import USB_VCP
        vcp = USB_VCP()
    except (ImportError, AttributeError, OSError):
        vcp = None
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
        print("CAMERA_INIT_ERROR", repr(error))
    if ENABLE_WATCHDOG:
        try:
            from machine import WDT
            watchdog = WDT(timeout=8000)
        except (ImportError, AttributeError, ValueError, OSError) as error:
            print("WATCHDOG_UNAVAILABLE", repr(error))
    print("V4 AUTO+INIT-ALIGN READY; AUTO_START", AUTO_START,
          "LIMITS", tracker.limits, "SEARCH", "OFF(WAIT_IN_PLACE)")
    now = time.ticks_ms()
    if AUTO_START:
        error = auto.start(now)
        if error:
            print("AUTO_START_ERROR", error)
    window = now
    usb_line = bytearray()
    while True:
        now = time.ticks_ms()
        if vcp and vcp.any():
            data = vcp.read(64)
            if data:
                for ch in data:
                    if ch in (10, 13):
                        if usb_line:
                            try:
                                handle_usb(tracker, auto, usb_line.decode(), now)
                            except Exception as error:
                                print("USB_CMD_ERROR", repr(error))
                            usb_line = bytearray()
                    else:
                        usb_line.append(ch)
        try:
            img = sensor.snapshot()
            target = detect(img)
            laser = detect_laser(img, tracker.aim)
            tracker.update(target, laser, now)
            auto.update(now)
            if target:
                img.draw_rectangle(target[2], color=(0, 255, 0), thickness=2)
                img.draw_cross(target[0], target[1], color=(255, 0, 0), size=12, thickness=2)
            if laser:
                img.draw_cross(laser[0], laser[1], color=(255, 0, 255), size=8, thickness=2)
                if target:
                    img.draw_line(laser[0], laser[1], target[0], target[1], color=(255, 0, 255))
            img.draw_cross(320, 240, color=(255, 255, 0), size=10)
            img.draw_string(4, 4, "%s %s" % (tracker.mode, tracker.reason), color=(0, 255, 0), scale=2)
            outputs = tracker.sent if tracker.servos else (0, 0)
            img.draw_string(4, 24, "PWM %d,%d" % (outputs[0], outputs[1]), color=(255, 255, 0), scale=2)
            label = "TARGET %d,%d" % target[:2] if target else "NO UNIQUE TARGET"
            img.draw_string(4, 44, label, color=(0, 255, 0), scale=2)
            lx, ly = laser if laser else (-1, -1)
            img.draw_string(4, 64, "LASER %d,%d ERR %d,%d" %
                            (lx, ly, int(round(tracker.aim_error[0])), int(round(tracker.aim_error[1]))),
                            color=(255, 0, 255), scale=2)
        except Exception as error:
            auto.cancel()
            tracker.off()
            tracker.reason = "CAMERA_OR_CONTROL_ERROR"
            print("VISION_ERROR; PWM OFF", repr(error))
        if time.ticks_diff(now, window) >= LOG_INTERVAL_MS:
            print("V4", tracker.mode, tracker.reason, "AIM", "LASER" if tracker.laser_aim else "CENTER",
                  "PWM", tracker.sent if tracker.servos else (0, 0),
                  "TARGET", tracker.target[:2] if tracker.target else None,
                  "LASER", tracker.laser)
            window = now
        if watchdog:
            watchdog.feed()
        time.sleep_ms(2)


if __name__ == "__main__":
    main()
