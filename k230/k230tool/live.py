# -*- coding: utf-8 -*-
"""
实时桥接（live）——让「人」和「AI Agent」都能随时看到 K230 的画面
=================================================================

后台常驻，把板端回传的 JPEG 落成文件：

    live/latest.jpg     最新一帧（默认已画好检测框和状态）
    live/latest.json    最新一帧的序号/帧率/检测结果/板端状态
    live/status.txt     一行文本状态（给 cat / 监控脚本用）
    live/history/       可选的历史帧（--history N）

Agent 直接读 latest.jpg 就能「看到」板子画面；读 latest.json 拿结构化检测结果。
停止：Ctrl-C、kill，或创建 live/STOP 文件。
"""

from __future__ import annotations
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

import argparse
import json
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

from .link import (
    ROOT, POSE_DEFAULT_CFG, K230Stream, describe_ports, find_port,
)

DEFAULT_OUT = ROOT / "live"
# 默认走 Pose：与 board/yolo_pose.py 的 CFG 单一来源（link.POSE_DEFAULT_CFG）
DEFAULT_MODEL = POSE_DEFAULT_CFG["model"]


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fp:
        fp.write(data)
        fp.flush()
        try:
            os.fsync(fp.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _labels(args) -> list:
    if args.labels_file:
        text = Path(args.labels_file).read_text(encoding="utf-8")
        text = text.strip()
        if text.startswith("["):
            return [str(x) for x in json.loads(text)]
        return [x.strip() for x in text.splitlines() if x.strip()]
    if args.labels:
        return [x.strip() for x in args.labels.split(",") if x.strip()]
    return list(POSE_DEFAULT_CFG["labels"])


def build_cfg(args) -> Dict[str, object]:
    labels = _labels(args)
    cfg: Dict[str, object] = {
        "w": args.width, "h": args.height, "q": args.quality,
        "fps": args.fps, "stats": 15, "limit": 0,
    }
    if not args.no_ai:
        cfg.update({
            "model": args.model or DEFAULT_MODEL,
            "labels": labels,
            "in_w": args.input_size, "in_h": args.input_size,
            "conf": args.conf, "cand_conf": 0.03,
            "far_refine": 0, "far_min": 100, "far_topk": 3, "far_max": 200,
            "low_refine": 1, "low_refine_topk": 2, "low_refine_mult": 7.0,
            "low_refine_max": 120, "low_refine_cooldown": 3,
            "dbg_save": args.dbg_save,
            "nms": args.nms,
            "kp_num": 5, "kp_dim": 3,
            "max_boxes": 20,
            "ai_every": args.ai_every,
            "jpeg_every": args.jpeg_every,
            "full_every": args.full_every,
            "roi_pad": args.roi_pad,
            "roi_min": args.roi_min,
            "roi_max": args.roi_max,
            # 回传 JPEG 质量与板端性能（见 board/k230_perf.py）；None = 用板端默认
            # 性能/编码键：没显式指定就不下发，沿用 board/k230_perf.py 的默认值
            "jpeg_q": args.jpeg_q,
            "enc_scale": args.enc_scale,
            "perf_mode": args.perf,
            "perf_fb": args.perf_fb,
            "reset_on_exit": 1,
            "uart_enable": 1, "uart_id": 3, "uart_tx": 32, "uart_rx": 33,
            "uart_baud": 115200,
        })
        # None = 没指定；不要让 None 覆盖板端默认值（板端会 int(None) 报错）
        for _k in ("jpeg_q", "enc_scale", "perf_mode", "perf_fb"):
            if cfg.get(_k) is None:
                del cfg[_k]
    return cfg


def _write_state(out: Path, state: Dict) -> None:
    body = json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write(out / "latest.json", body)
    line = "seq=%s fps=%.1f targets=%d age=%.0fms connected=%s" % (
        state.get("seq", "-"), state.get("fps", 0.0),
        len(state.get("detections") or []),
        state.get("age_ms") or 0.0, state.get("connected"))
    _atomic_write(out / "status.txt", (line + "\n").encode("utf-8"))


class Live:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop_file = self.out / "STOP"
        self.cfg = build_cfg(args)
        self.labels = _labels(args)
        self.stop = False
        self.stream: Optional[K230Stream] = None
        self.last_seq = 0
        self.last_frame_ts = 0.0
        self.frame_times: Deque[float] = deque(maxlen=30)
        self.total = 0
        self._last_geo = {}          # 最近一次带 JPEG 的 roi/jpeg 几何
        self.history_saved = 0
        self.last_history = 0.0
        self.last_state = 0.0
        self.started = time.time()
        self.log = (lambda *a: None) if args.quiet else (
            lambda msg: print("[live] %s" % msg, file=sys.stderr, flush=True))

    # ------------------------------------------------------------ 信号
    def _install_signals(self):
        def _stop(_sig, _frm):
            self.stop = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _stop)
            except Exception:
                pass

    def _check_stop(self) -> bool:
        if self.stop:
            return True
        if self.stop_file.exists():
            self.log("检测到 %s，停止" % self.stop_file)
            try:
                self.stop_file.unlink()
            except OSError:
                pass
            return True
        if self.args.duration and time.time() - self.started >= self.args.duration:
            self.log("到达 --duration %.0fs，停止" % self.args.duration)
            return True
        return False

    # ------------------------------------------------------------ 状态
    def _state(self, connected: bool = True) -> Dict:
        now = time.time()
        st = self.stream
        target = st.detections(self.last_seq) if (st and not self.args.no_ai) else None
        geo = self._geo(target)
        dets = []
        bull = None
        kpts = None
        if target:
            names = target.get("labels") or self.labels
            if target.get("bull"):
                bull = target["bull"]
            kpts = target.get("kp")
            for item in target.get("det") or []:
                try:
                    cls = int(item[5])
                except (TypeError, ValueError, IndexError):
                    cls = 0
                dets.append({
                    "label": names[cls] if cls < len(names) else "cls%d" % cls,
                    "cls": cls,
                    "conf": round(float(item[4]), 3),
                    "box": [int(round(v)) for v in item[:4]],
                })
        return {
            "source": "k230-usb",
            "ts": round(now, 3),
            "iso_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seq": self.last_seq,
            "age_ms": round((now - self.last_frame_ts) * 1000.0, 1) if self.last_frame_ts else None,
            "fps": round(self._fps(), 2),
            "size": [geo.get("w") or self.args.width, geo.get("h") or self.args.height],
            "jpeg_bytes": None,
            "connected": connected,
            "total_frames": self.total,
            "ai": not self.args.no_ai,
            "model": None if self.args.no_ai else (self.args.model or DEFAULT_MODEL),
            "labels": self.labels,
            "detections": dets,
            "bull": bull,
            "kpts": kpts,
            "roi": geo.get("roi"),
            "jpeg": geo.get("jpeg"),
            "infer_ms": target.get("ms") if target else None,
            "s1_ms": target.get("s1") if target else None,
            "s2_ms": target.get("s2") if target else None,
            "jpg_ms": target.get("jpg") if target else None,
            "loop_ms": target.get("loop") if target else None,
            "uart_ms": target.get("uart") if target else None,
            "post_ms": target.get("post") if target else None,
            "refine_ms": target.get("ref_ms") if target else None,
            "board": dict(st.stats) if st else {},
            "hello": dict(st.hello) if st else {},
        }

    def _fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        dt = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / dt if dt > 0 else 0.0

    # ------------------------------------------------------------ 帧处理
    def _save_frame(self, jpeg: bytes, seq: int, target) -> None:
        from .decode import annotate

        annotated = jpeg
        if not self.args.no_overlay:
            try:
                annotated = annotate(
                    jpeg, target, self.labels,
                    fps=self._fps(), seq=seq,
                    extra="model %s" % Path(self.cfg.get("model", "?")).name
                    if not self.args.no_ai else None,
                )
            except Exception as exc:       # opencv 缺失时退回原始帧
                self.log("叠加失败（%s），写原始帧" % exc)
        _atomic_write(self.out / "latest.jpg", annotated)

        if self.args.history and (time.time() - self.last_history >= self.args.history_every):
            self.last_history = time.time()
            hist = self.out / "history"
            hist.mkdir(exist_ok=True)
            self.history_saved += 1
            _atomic_write(hist / ("frame_%06d.jpg" % seq), annotated)
            files = sorted(hist.glob("frame_*.jpg"))
            for old in files[:-self.args.history]:
                try:
                    old.unlink()
                except OSError:
                    pass

    def _apply_target(self, target) -> None:
        """实时把板端每帧里的 roi / jpeg 尺寸同步到 state。

        ROI 模式下图像回传是隔帧的（jpeg_every=2），若只在写 latest.json 时从
        最新目标取，约一半概率拿到"本帧没有 JPEG"的那条 -> roi/jpeg=None ->
        叠加层按全图缩放，框/靶心会整体错位。这里用最近一次有 JPEG 的目标。
        """
        if not isinstance(target, dict):
            return
        geo = {k: target.get(k) for k in ("w", "h", "roi", "jpeg")}
        if geo["roi"] is not None and geo["jpeg"] is not None:
            self._last_geo = geo

    def _geo(self, target):
        geo = dict(self._last_geo or {})
        if isinstance(target, dict):
            for k in ("w", "h", "roi", "jpeg", "labels", "task"):
                if target.get(k) is not None:
                    geo[k] = target.get(k)
        return geo

    def _write(self, connected: bool = True):
        state = self._state(connected)
        if self.last_seq and (self.out / "latest.jpg").exists():
            try:
                state["jpeg_bytes"] = (self.out / "latest.jpg").stat().st_size
            except OSError:
                pass
        _write_state(self.out, state)
        self.last_state = time.time()

    # ------------------------------------------------------------ 主循环
    def run(self) -> int:
        self._install_signals()
        if self.stop_file.exists():
            try:
                self.stop_file.unlink()
            except OSError:
                pass
        if not find_port():
            print("没有找到 K230。当前串口：\n%s" % describe_ports(), file=sys.stderr)
            self._write(connected=False)
            return 2

        self.log("输出目录 %s" % self.out.resolve())
        self.log("启动 %s 流（%dx%d q%d）%s" % (
            "YOLOv8" if not self.args.no_ai else "原始",
            self.args.width, self.args.height, self.args.quality,
            "" if self.args.no_ai else "model=%s" % self.cfg.get("model")))
        self.stream = K230Stream(
            cfg=self.cfg, port=self.args.port, ai=not self.args.no_ai,
            auto_restart=True, verbose=self.args.verbose,
            log=lambda msg: self.log(msg))
        self.stream.start(timeout=self.args.timeout)
        if not self.stream.hello:
            self.log("板端无响应：%s" % (self.stream.last_error or "超时"))
        else:
            self.log("已连接：w=%s h=%s model=%s" % (
                self.stream.hello.get("w"), self.stream.hello.get("h"),
                self.stream.hello.get("model", "-")))

        last_id = 0
        try:
            while not self._check_stop():
                jpeg, idx = self.stream.wait_frame(last_id, timeout=0.5)
                if jpeg is not None and idx != last_id:
                    last_id = idx
                    self.last_seq = idx
                    now = time.time()
                    self.last_frame_ts = now
                    self.frame_times.append(now)
                    self.total += 1
                    target = (self.stream.detections(idx)
                              if not self.args.no_ai else None)
                    self._apply_target(target)
                    self._save_frame(jpeg, idx, target)
                    if now - self.last_state >= self.args.json_every:
                        self._write(connected=True)
                else:
                    # 没有新帧：更新心跳/掉线状态
                    if time.time() - self.last_state >= self.args.json_every:
                        connected = bool(self.stream and self.stream.connected)
                        stale = (self.last_frame_ts and
                                 time.time() - self.last_frame_ts > self.args.stale_after)
                        self._write(connected=connected and not stale)
        finally:
            self.log("正在停止板端脚本…")
            try:
                self.stream.stop()
            except Exception as exc:
                self.log("停止异常：%s" % exc)
            self._write(connected=False)
            self.log("已停止：共 %d 帧，输出 %s" % (self.total, self.out.resolve()))
        return 0


# ---------------------------------------------------------------- 快照
def snapshot(args) -> int:
    """取一张图：优先复用正在运行的 live/latest.jpg，否则临时开一次流。"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    live_dir = Path(args.live_dir)
    latest = live_dir / "latest.jpg"
    if not args.no_reuse and latest.exists():
        age = time.time() - latest.stat().st_mtime
        if age <= args.max_age:
            data = latest.read_bytes()
            _atomic_write(out, data)
            try:
                meta_src = live_dir / "latest.json"
                if meta_src.exists():
                    Path(str(out) + ".json").write_bytes(meta_src.read_bytes())
            except OSError:
                pass
            print("[snapshot] 复用 %s（%.1fs 前）-> %s" % (latest, age, out))
            return 0

    if not find_port():
        print("没有找到 K230。当前串口：\n%s" % describe_ports(), file=sys.stderr)
        return 2
    cfg = build_cfg(args)
    labels = _labels(args)
    st = K230Stream(cfg=cfg, port=args.port, ai=not args.no_ai,
                    auto_restart=False, log=lambda m: print("[snapshot]", m))
    st.start(timeout=args.timeout)
    if not st.hello:
        print("[snapshot] 板端无响应：%s" % (st.last_error or "超时"), file=sys.stderr)
        st.stop()
        return 3
    jpeg, idx = st.wait_frame(0, timeout=args.timeout)
    if jpeg is None:
        print("[snapshot] 没有收到帧", file=sys.stderr)
        st.stop()
        return 4
    target = st.detections(idx) if not args.no_ai else None
    annotated = jpeg
    if not args.no_overlay:
        from .decode import annotate
        try:
            annotated = annotate(jpeg, target, labels, fps=0.0, seq=idx)
        except Exception as exc:
            print("[snapshot] 叠加失败（%s），保存原始帧" % exc, file=sys.stderr)
    _atomic_write(out, annotated)
    meta = {
        "seq": idx, "size": [args.width, args.height],
        "target": target, "hello": st.hello, "stats": st.stats,
        "saved": str(out),
    }
    try:
        Path(str(out) + ".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    print("[snapshot] seq=%s targets=%d -> %s" % (
        idx, len(target.get("det") or []) if target else 0, out))
    st.stop()
    return 0


# ---------------------------------------------------------------- 状态
def status(args) -> int:
    live_dir = Path(args.live_dir)
    payload: Dict = {"live_dir": str(live_dir)}
    for name in ("latest.json", "status.txt"):
        p = live_dir / name
        if p.exists():
            payload[name] = p.read_text(encoding="utf-8", errors="replace").strip()
            payload[name + "_age_s"] = round(time.time() - p.stat().st_mtime, 1)
    port = find_port()
    payload["port"] = port

    # live 正在占用串口时 find_port() 会失败，此时以 latest.json 的新鲜度为准
    fresh_live = False
    try:
        meta = json.loads((live_dir / "latest.json").read_text(encoding="utf-8"))
        age = time.time() - (live_dir / "latest.json").stat().st_mtime
        fresh_live = bool(meta.get("connected")) and age < 5.0
        payload["live_connected"] = meta.get("connected")
        payload["live_alive"] = fresh_live
    except Exception:
        pass
    payload["port_ok"] = bool(port) or fresh_live
    payload["port_held_by_live"] = bool(fresh_live and not port)

    models = sorted((ROOT / "models").glob("*.kmodel")) if (ROOT / "models").exists() else []
    payload["local_models"] = [p.name for p in models]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["port_ok"] or (live_dir / "latest.jpg").exists() else 1


# ---------------------------------------------------------------- CLI
def _add_live_args(p):
    p.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    p.add_argument("--port", default=None)
    p.add_argument("--duration", type=float, default=0, help="运行秒数，0=一直跑")
    p.add_argument("--w", "--width", dest="width", type=int, default=640)
    p.add_argument("--h", "--height", dest="height", type=int, default=480)
    p.add_argument("--aw", type=int, default=0, help="AI 输入宽，默认同 --w")
    p.add_argument("--ah", type=int, default=0)
    p.add_argument("--q", "--quality", dest="quality", type=int, default=70)
    p.add_argument("--fps", type=int, default=0, help="板端限速，0=跑满")
    p.add_argument("--model", default=None, help="板端 kmodel 路径")
    p.add_argument("--labels", default=None, help="逗号分隔，如 target")
    p.add_argument("--labels-file", default=None, help="标签文件（每行一个或 JSON）")
    p.add_argument("--input-size", type=int, default=384, help="kmodel 输入尺寸")
    p.add_argument("--jpeg-q", type=int, default=None,
                   help="回传 JPEG 质量（默认沿用板端 --q；图像 ROI 建议 88~92）")
    p.add_argument("--enc-scale", type=float, default=None,
                   help="回传图缩放（默认沿用板端 0.80 -> 512x384；1.0 = 原生，帧率减半）")
    p.add_argument("--perf", choices=("auto", "balanced", "performance"), default=None,
                   help="板端 CPU 频率档位（默认沿用板端配置，通常 performance）")
    p.add_argument("--perf-fb", type=int, default=None,
                   help="相机帧缓冲个数（3 更稳，内存紧张用 0 = 固件默认）")
    p.add_argument("--ai-every", type=int, default=2,
                   help="每 N 帧跑一次 Pose 推理（1=每帧，2=隔帧，省一半推理）")
    p.add_argument("--jpeg-every", type=int, default=2,
                   help="每 N 帧回传一张图（结果 JSON 每帧都发）")
    p.add_argument("--full-every", type=int, default=30,
                   help="每 N 帧回传一张缩小全图")
    p.add_argument("--roi-pad", type=float, default=0.25)
    p.add_argument("--roi-min", type=int, default=160)
    p.add_argument("--roi-max", type=int, default=256)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.45)
    p.add_argument("--dbg-save", type=int, default=0,
                   help="每 N 帧把画好的整帧存 /sdcard/pose_dbg.jpg（0=关）")
    p.add_argument("--no-ai", action="store_true", help="只回传画面，不跑 YOLO")
    p.add_argument("--no-overlay", action="store_true", help="不做 PC 端画框")
    p.add_argument("--history", type=int, default=0, help="保留最近 N 张历史帧")
    p.add_argument("--history-every", type=float, default=5.0)
    p.add_argument("--json-every", type=float, default=0.25)
    p.add_argument("--stale-after", type=float, default=10.0)
    p.add_argument("--timeout", type=float, default=40.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def add_parser(sub):
    return _add_live_args(sub.add_parser("live", help="实时桥接：写 live/latest.jpg"))


def build_parser():
    ap = argparse.ArgumentParser(prog="k230tool live", description="K230 实时桥接")
    return _add_live_args(ap)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "live":
        argv = argv[1:]
    return Live(build_parser().parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
