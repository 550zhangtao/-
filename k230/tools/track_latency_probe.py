# -*- coding: utf-8 -*-
"""track_latency_probe.py —— 跟踪链路延迟探针（PC 侧，零板端改动）

同时看两只 USB 串口：
  * K230 (/dev/ttyACM0)：读 T_TARGET 遥测（板端 seq / ms / jpg / loop / bull）
  * ESP32 (/dev/ttyUSB0)：轮询 status，拿 frames / age / ctrlErr / pan / tilt

产出：
  * 控制量（$KT）实际更新间隔的中位数/P95 —— 决定"预测时要补多少帧"
  * 板端 AI / 编码 / loop 耗时 —— 确认瓶颈在推理还是编码
  * ESP32 侧看到的 `age`（数据年龄）—— 近似"拍照→控制"延迟
  * 一段 10Hz 的 (seq, bull, ctrlErr) 时间线，可直接对照 ESP32 的响应

用法：
    python3 -m tools.track_latency_probe --k230 /dev/ttyACM0 \\
        --esp32 /dev/ttyUSB0 --seconds 20

注意：会占用 K230 串口推流（与 CanMV IDE 互斥，见 AGENTS 第 2 节）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # k230/
sys.path.insert(0, str(ROOT))

from k230tool.link import K230Stream  # noqa: E402


def _pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def _stats(xs):
    if not xs:
        return "n=0"
    return ("n=%d 中位=%.1f 均值=%.1f P95=%.1f 最大=%.1f"
            % (len(xs), statistics.median(xs), statistics.fmean(xs),
               _pct(xs, 95), max(xs)))


class Esp32Probe(threading.Thread):
    """后台 10Hz 读 ESP32 status，记录 (t, frames, age, ctrlErr, pan, tilt)。"""

    def __init__(self, dev, period=0.1):
        super().__init__(daemon=True)
        self.dev = dev
        self.period = period
        self.samples = []          # [(t, dict)]
        self.t_start = time.monotonic()
        self.stop_flag = threading.Event()
        self.error = None
        self.ser = None

    def open(self):
        import serial
        self.ser = serial.Serial(self.dev, 115200, timeout=0.05)
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    def run(self):
        try:
            if self.ser is None:
                self.open()
            # 先切到 TRACK；若 ESP32 因丢靶自动回 NONE，前 5s 内每隔 0.5s 重新拉起
            self.ser.write(b"gtrack\n")
            pulled = 0
            while not self.stop_flag.is_set():
                t = time.monotonic()
                self.ser.write(b"status\n")
                t0 = time.monotonic()
                buf = b""
                while time.monotonic() - t0 < 0.25:
                    buf += self.ser.read(4096)
                text = buf.decode("utf-8", "replace")
                for line in text.replace("\r", "\n").splitlines():
                    line = line.strip()
                    if line.startswith("PAN="):
                        d = self._parse(line)
                        if d:
                            self.samples.append((t, d))
                            if (d["state"] == 0 and pulled < 10
                                    and t - self.t_start < 5.0):
                                pulled += 1
                                self.ser.write(b"gtrack\n")
                        break
                rest = self.period - (time.monotonic() - t)
                if rest > 0:
                    self.stop_flag.wait(rest)
        except Exception as e:                      # pragma: no cover
            self.error = repr(e)

    @staticmethod
    def _parse(line):
        out = {}
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                out[k] = v
        try:
            return {
                "frames": int(out.get("frames", -1)),
                "age": int(out.get("age", -1)),
                "ctrlErr": out.get("ctrlErr", ""),
                "pan": int(out.get("pan", -1)),
                "tilt": int(out.get("tilt", -1)),
                "state": int(out.get("state", -1)),
                "bull": out.get("bull", ""),
            }
        except Exception:
            return None

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2.0)
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k230", default="/dev/ttyACM0")
    ap.add_argument("--esp32", default="/dev/ttyUSB0")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default=str(ROOT / "live" / "latency_probe.json"))
    args = ap.parse_args(argv)

    stream = K230Stream(port=args.k230, ai=True, verbose=False,
                        log=lambda m: print("[k230]", m))
    esp = Esp32Probe(args.esp32)
    print("[probe] 启动：K230 推流 + ESP32 10Hz 采状态，%ds" % args.seconds)
    try:
        stream.start(timeout=25)
        esp.start()
        t0 = time.monotonic()
        seen = -1
        beats = []          # [(t, seq, bull, conf)]
        while time.monotonic() - t0 < args.seconds:
            with stream._lock:                       # noqa: SLF001
                seq = stream.target_seq
                tgt = dict(stream.target) if stream.target else None
            if tgt is not None and seq != seen:
                seen = seq
                beats.append((round(time.monotonic() - t0, 3), seq,
                              tgt.get("bull"), tgt.get("ms"), tgt.get("jpg"),
                              tgt.get("loop"), tgt.get("ai")))
            time.sleep(0.01)
    finally:
        esp.stop()
        try:
            stream.stop()
        except Exception:
            pass

    # ---- 分析 ----
    seqs = [b[1] for b in beats]
    gaps = [round((b2[0] - b1[0]) * 1000.0, 1)
            for b1, b2 in zip(beats, beats[1:]) if b2[1] > b1[1]]
    ai_ms = [b[3] for b in beats if isinstance(b[3], (int, float)) and b[3] > 0]
    jpg_ms = [b[4] for b in beats if isinstance(b[4], (int, float)) and b[4] > 0]
    loop_ms = [b[5] for b in beats if isinstance(b[5], (int, float)) and b[5] > 0]

    esp_samples = [s for s in esp.samples]
    ages = []
    frames = []
    for _t, d in esp_samples:
        if d["age"] >= 0:
            ages.append(float(d["age"]))
        if d["frames"] >= 0:
            frames.append(d["frames"])

    print("\n=== K230 → PC 遥测（%d 帧） ===" % len(beats))
    print("  $KT/遥测间隔 ms :", _stats(gaps))
    print("  板端 infer ms   :", _stats(ai_ms))
    print("  板端 jpg   ms   :", _stats(jpg_ms))
    print("  板端 loop  ms   :", _stats(loop_ms))
    if seqs:
        print("  seq 范围        : %d .. %d（%.1f 帧/秒）"
              % (min(seqs), max(seqs),
                 (max(seqs) - min(seqs)) / max(1e-6, beats[-1][0] - beats[0][0])))

    print("\n=== ESP32 侧（%d 次采样，10Hz） ===" % len(esp_samples))
    if ages:
        fresh = [a for a in ages if a < 5000]
        print("  age(ms) 全量    :", _stats(ages))
        print("  age(ms) 新鲜帧  :", _stats(fresh))
    if len(frames) > 1:
        df = frames[-1] - frames[0]
        dt = esp_samples[-1][0] - esp_samples[0][0]
        print("  ESP32 实际收帧率: %.2f Hz（%d 帧 / %.1fs）"
              % (df / max(1e-6, dt), df, dt))
    if esp.error:
        print("  ESP32 读取错误  :", esp.error)

    print("\n=== 时间线（前 30 个控制拍）===")
    print("  t(s)   seq    bull(x,y)        ms  jpg  loop")
    for b in beats[:30]:
        print("  %5.2f  %5d  %-16s %4s %4s %5s"
              % (b[0], b[1], str(b[2]), b[3], b[4], b[5]))

    payload = {
        "beats": beats,
        "esp32": [(round(t - (beats[0][0] if beats else 0), 3), d)
                  for t, d in esp_samples],
        "summary": {
            "kt_gap_ms": gaps,
            "infer_ms": ai_ms,
            "jpg_ms": jpg_ms,
            "loop_ms": loop_ms,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("\n[probe] 原始数据 -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
