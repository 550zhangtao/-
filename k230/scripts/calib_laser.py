#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calib_laser.py —— 激光光斑视觉标定（PC 端，ON/OFF 主动差分）
=============================================================

为什么在 PC 上做标定
--------------------
板端 `find_blobs` 实测有 ~12ms 固定开销（与 ROI 大小基本无关），
而标定只需要**做一次**，放在 PC 上可以用 OpenCV 做最稳的那套。

当前硬件：**激光常亮，GPIO5 已弃用**（2026-10-22 用户确认），所以默认
走"单帧 LAB + 人工点选"：
    抓一帧 → LAB 红点候选 → 弹出窗口用鼠标点一下光斑 → 写 laser_calib.json
这一步只需要一次，人工点选比任何阈值都可靠。

如果以后激光接到可控开关（或 ESP32 GPIO 恢复），可以加 `--pair`：
    |ON - OFF| 帧差 → 阈值 → 形态学 → 连通域 → 亚像素质心
这是主动照明（lock-in）标准做法：背景/红物/环境光全部抵消，最稳，
但要求能可靠地关掉激光。

用法
----
1) 抓一帧标定（默认，激光常亮；需要 K230 USB，云台保持不动）：

    cd k230
    python3 scripts/calib_laser.py --grab --push

2) 已经有一张图：

    python3 scripts/calib_laser.py --image on.jpg --push

3) 手动给坐标（坐标是 640x480 原图）：

    python3 scripts/calib_laser.py --x 408 --y 278 --push

4) 可控激光时做 ON/OFF 主动差分（更稳）：

    python3 scripts/calib_laser.py --grab --pair --esp-port /dev/ttyUSB0 --push

5) 多距离拟合 `spot = c + k·s`（s=靶纸表观边长，px）：

    python3 scripts/calib_laser.py --fit points.json
    # points.json = [{"x":408,"y":278,"s":42}, {"x":392,"y":271,"s":78}]

输出
----
默认写 `k230/live/laser_calib.json`，`--push` 再推到板端
`/sdcard/laser_calib.json`。板端 `board/laser_aim.py` 启动时读它。

注意
----
- 抓图会占用 K230 串口；先 `bash scripts/mode.sh debug`（工作模式下 REPL 被代理占用）；
- 同一串口不能同时给 `k230tool live` / CanMV IDE 用；
- 标定时光斑要落在画面内、云台不要动；换距离/角度后视觉上确认一次。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("需要 opencv-python：pip3 install opencv-python")

ROOT = Path(__file__).resolve().parent.parent          # k230/
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "live" / "laser_calib.json"
BOARD_CALIB = "/sdcard/laser_calib.json"


# ------------------------------------------------------------------ 基础
def _log(msg: str) -> None:
    print("[calib] %s" % msg, flush=True)


def load_img(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("读不到图片：%s" % path)
    return img


def _scale_to_coord(x: float, y: float, img_w: int, img_h: int,
                    cw: int, ch: int):
    return (x * cw / float(img_w), y * ch / float(img_h))


# ------------------------------------------------------------------ 差分检测
def detect_diff(on: np.ndarray, off: np.ndarray, thresh: int = 25,
                min_area: int = 4, max_area: int = 4000,
                max_side: int = 120):
    """|ON-OFF| 差分 + 连通域，返回候选列表（按 score 降序）。

    候选：{x,y,area,w,h,peak,mean,score}
    """
    if on.shape != off.shape:
        raise SystemExit("ON/OFF 尺寸不一致：%s vs %s" % (on.shape, off.shape))
    diff = cv2.absdiff(on, off).astype(np.float32)
    gray = diff.max(axis=2)
    mask = (gray >= int(thresh)).astype(np.uint8) * 255
    # 不做 open：真实 5~10px 光斑的阈值环只有 2~3px 宽，3x3 open 会把它抹掉
    # （实测板端激光 a*≈163、环 19px，open 后 0px）。只做 close 把环连起来。
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < min_area or area > max_area:
            continue
        if w > max_side or h > max_side:
            continue
        ys, xs = np.where(lab[y:y + h, x:x + w] == i)
        if len(xs) == 0:
            continue
        wgt = gray[y:y + h, x:x + w][ys, xs]
        sw = float(wgt.sum())
        if sw <= 0:
            continue
        cx = float((xs * wgt).sum() / sw) + x
        cy = float((ys * wgt).sum() / sw) + y
        compact = area / float(max(1, w * h))
        peak = float(wgt.max())
        mean = float(wgt.mean())
        # 差分里最像一个"点"的：亮、紧凑、面积别太大
        score = peak * 1.0 + mean * 0.5 + compact * 40.0 - area * 0.02
        out.append({"x": cx, "y": cy, "area": area, "w": w, "h": h,
                    "peak": peak, "mean": mean, "compact": compact,
                    "score": score})
    out.sort(key=lambda c: -c["score"])
    return out, mask


# ------------------------------------------------------------------ 单帧兜底
def detect_single(img: np.ndarray, a_min: int = 150, l_min: int = 40,
                  min_area: int = 3, max_area: int = 800,
                  max_side: int = 80):
    """单帧 LAB 红点检测（兜底）。a_min=165 约等于 a*≈37（OpenCV 偏移 128）。

    排序用 **a_max** 而不是面积：真激光核心 a* 能到 200+（实测 205），
    红色杂物一般 160~185；面积只作为弱约束。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ll, a, _b = cv2.split(lab)
    mask = ((a >= int(a_min)) & (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) >= int(l_min)))
    mask = (mask.astype(np.uint8) * 255)
    # 不做 open（会把 5~10px 光斑的细阈值环抹掉），只 close 连成一块；
    # 5x5 是为了把被 JPEG 压断成 2~3 块的光斑重新合成一个连通域
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    bb, gg, rr = cv2.split(img.astype(np.int16))
    mn = np.minimum(np.minimum(bb, gg), rr)
    n, lab_img, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < min_area or area > max_area:
            continue
        if w > max_side or h > max_side:
            continue
        ys, xs = np.where(lab_img[y:y + h, x:x + w] == i)
        if len(xs) == 0:
            continue
        av = a[y:y + h, x:x + w][ys, xs].astype(np.float32)
        wt = np.clip(av - 128.0, 0.0, None)
        sw = float(wt.sum())
        if sw <= 0:
            continue
        cx = float((xs * wt).sum() / sw) + x
        cy = float((ys * wt).sum() / sw) + y
        amax = float(av.max())
        amean = float(av.mean())
        compact = area / float(max(1, w * h))
        # 过曝白核：激光中心常被曝光打成 L≈255、三通道都>=200；
        # 红色印刷品/电缆没有这个特征。它只参与排序，不参与阈值。
        hot = int((((ll[y:y + h, x:x + w][ys, xs] >= 240) &
                    (mn[y:y + h, x:x + w][ys, xs] >= 200))).sum())
        score = ((amax - 128.0) * 3.0 + compact * 20.0 - area * 0.02
                 + (40.0 if hot >= 1 else 0.0))
        out.append({"x": cx, "y": cy, "area": area, "w": w, "h": h,
                    "peak": amax, "mean": amean, "compact": compact,
                    "score": score})
    out.sort(key=lambda c: -c["score"])
    return out, mask


# ------------------------------------------------------------------ GUI
class _Picker:
    """点选 + 确认。Enter/Space 接受，鼠标左键改点，r 重跑，q 退出。"""

    def __init__(self, title: str):
        self.title = title
        self.pt = None
        self.done = False
        self.rerun = False
        self.quit = False

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pt = (float(x), float(y))

    def run(self, view: np.ndarray, init_pt=None):
        self.pt = tuple(init_pt) if init_pt is not None else None
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.title, self._on_mouse)
        while True:
            canvas = view.copy()
            if self.pt is not None:
                x, y = int(round(self.pt[0])), int(round(self.pt[1]))
                cv2.drawMarker(canvas, (x, y), (0, 255, 255),
                               cv2.MARKER_CROSS, 24, 2)
                cv2.circle(canvas, (x, y), 14, (0, 255, 255), 2)
                cv2.putText(canvas, "click to adjust, Enter=OK, r=redo, q=quit",
                            (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(self.title, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10, 32):          # Enter / Space
                self.done = True
                break
            if key in (ord("q"), 27):
                self.quit = True
                break
            if key == ord("r"):
                self.rerun = True
                break
        cv2.destroyWindow(self.title)
        return self.pt


def _annotate(img: np.ndarray, cands, chosen: int = 0) -> np.ndarray:
    out = img.copy()
    for i, c in enumerate(cands[:5]):
        x, y = int(round(c["x"])), int(round(c["y"]))
        color = (0, 255, 0) if i == chosen else (128, 128, 128)
        cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, 18, 2)
        cv2.putText(out, "#%d %.0f" % (i + 1, c["score"]),
                    (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    cv2.LINE_AA)
        cv2.rectangle(out, (x - 12, y - 12), (x + 12, y + 12), color, 1)
    return out


# ------------------------------------------------------------------ 抓一对 ON/OFF
def _esp_send(ser, cmd: str) -> None:
    ser.write((cmd + "\n").encode())
    ser.flush()
    time.sleep(0.35)


def grab_frames(port: str, pair: bool = False, esp_port: str = None,
                manual: bool = False, quality: int = 95):
    """抓图。pair=False 返回 (on, None)；pair=True 返回 (on, off)。"""
    from k230tool.link import K230Stream, find_port

    if not find_port(port):
        raise SystemExit("没有找到 K230；先检查 USB 和 `bash scripts/mode.sh debug`")

    esp = None
    if pair and not manual:
        try:
            import serial
        except ImportError:
            raise SystemExit("需要 pyserial")
        try:
            esp = serial.Serial(esp_port, 115200, timeout=0.3)
            time.sleep(0.2)
            esp.reset_input_buffer()
            _log("ESP32：%s（laserpin 1=灭，0=亮；GPIO5 弃用时此路无效）" % esp_port)
        except Exception as e:
            raise SystemExit("打不开 ESP32 %s：%s" % (esp_port, e))

    cfg = {"w": 640, "h": 480, "q": int(quality), "fps": 0, "stats": 0}
    with K230Stream(cfg=cfg, port=port, log=lambda m: _log(m)) as st:
        _log("等 K230 出图…")
        idx = -1
        for _ in range(3):
            jpeg, idx = st.wait_frame(idx, timeout=8)
        if jpeg is None:
            raise SystemExit("K230 没有回帧")

        if not pair:
            if manual:
                print("\n>>> 确保激光亮着、云台不要动，然后按回车抓图…", flush=True)
                try:
                    input()
                except EOFError:
                    pass
            on_jpeg, idx = st.wait_frame(idx, timeout=8)
            on = cv2.imdecode(np.frombuffer(on_jpeg, np.uint8), cv2.IMREAD_COLOR)
            if on is None:
                raise SystemExit("抓图失败")
            _log("抓到 %s（激光常亮模式）" % (on.shape,))
            return on, None

        if manual:
            print("\n>>> 请先【关掉激光】（物理开关/断电），云台不要动，"
                  "然后按回车…", flush=True)
            try:
                input()
            except EOFError:
                pass
        else:
            _esp_send(esp, "laserpin 1")
        time.sleep(0.6)
        off_jpeg, idx = st.wait_frame(idx, timeout=8)
        off = cv2.imdecode(np.frombuffer(off_jpeg, np.uint8), cv2.IMREAD_COLOR)

        if manual:
            print(">>> 现在【打开激光】，云台不要动，按回车…", flush=True)
            try:
                input()
            except EOFError:
                pass
        else:
            _esp_send(esp, "laserpin 0")
        time.sleep(0.6)
        on_jpeg, idx = st.wait_frame(idx, timeout=8)
        on = cv2.imdecode(np.frombuffer(on_jpeg, np.uint8), cv2.IMREAD_COLOR)

    if esp is not None:
        try:
            esp.close()
        except Exception:
            pass
    if off is None or on is None:
        raise SystemExit("抓图失败")
    _log("抓到 ON %s / OFF %s" % (on.shape, off.shape))
    return on, off


# ------------------------------------------------------------------ 拟合/输出
def fit_points(points):
    """points=[{x,y,s}]；拟合 x=cx+kx*(s-ref), y=cy+ky*(s-ref)。"""
    if len(points) < 2:
        raise SystemExit("--fit 至少需要 2 个点")
    s = np.array([float(p["s"]) for p in points], np.float64)
    xs = np.array([float(p["x"]) for p in points], np.float64)
    ys = np.array([float(p["y"]) for p in points], np.float64)
    ref = float(s.mean())
    ds = s - ref
    den = float((ds * ds).sum())
    if den <= 1e-9:
        raise SystemExit("--fit 的 s 都相同，无法拟合 k")
    kx = float((ds * (xs - xs.mean())).sum() / den)
    ky = float((ds * (ys - ys.mean())).sum() / den)
    return {"x": float(xs.mean()), "y": float(ys.mean()),
            "kx": kx, "ky": ky, "ref": ref,
            "method": "fit", "n": len(points),
            "resid": float(np.sqrt(((xs - (xs.mean() + kx * ds)) ** 2 +
                                    (ys - (ys.mean() + ky * ds)) ** 2)).mean())}


def write_calib(out: Path, x: float, y: float, kx: float = 0.0,
                ky: float = 0.0, ref: float = 0.0, method: str = "diff",
                extra=None):
    out.parent.mkdir(parents=True, exist_ok=True)
    d = {"x": round(float(x), 2), "y": round(float(y), 2),
         "kx": round(float(kx), 5), "ky": round(float(ky), 5),
         "ref": round(float(ref), 2), "method": method,
         "ts": int(time.time())}
    if extra:
        d.update(extra)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(out)
    _log("已写 %s：%s" % (out, d))
    return d


def push_calib(local: Path, port=None):
    from k230tool.link import K230Link
    from k230tool.push_file import push_one

    lk = K230Link(port=port)
    lk.open()
    try:
        ok = push_one(lk, str(local), BOARD_CALIB, verbose=True)
        if not ok:
            raise SystemExit("推送 %s 失败" % BOARD_CALIB)
        _log("已推到板端 %s（重新 START/上电后生效）" % BOARD_CALIB)
    finally:
        lk.close()


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="激光光斑视觉标定（ON/OFF 差分）")
    ap.add_argument("--on", help="激光 ON 的单帧图片")
    ap.add_argument("--off", help="激光 OFF 的单帧图片（与 --on 同机位）")
    ap.add_argument("--image", help="只有一张 ON 图时用（LAB 单帧兜底）")
    ap.add_argument("--x", type=float, help="手动光斑 x（640x480 原图坐标）")
    ap.add_argument("--y", type=float, help="手动光斑 y")
    ap.add_argument("--fit", help="多点拟合 JSON： [{x,y,s}, ...]")
    ap.add_argument("--grab", action="store_true",
                    help="从板子抓图（默认抓一帧；--pair 时抓 ON/OFF）")
    ap.add_argument("--pair", action="store_true",
                    help="ON/OFF 主动差分（需要能可靠关掉激光）")
    ap.add_argument("--port", default=None, help="K230 串口，默认自动")
    ap.add_argument("--esp-port", default=None,
                    help="ESP32 串口（如 /dev/ttyUSB0），不填则手动开关激光")
    ap.add_argument("--manual", action="store_true",
                    help="手动开关激光（每步按回车），不用 ESP32 串口")
    ap.add_argument("--no-gui", action="store_true", help="无窗口，自动接受最佳候选")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出 JSON")
    ap.add_argument("--push", action="store_true", help="推到 /sdcard/laser_calib.json")
    ap.add_argument("--coord-w", type=int, default=640, help="输出坐标宽")
    ap.add_argument("--coord-h", type=int, default=480, help="输出坐标高")
    ap.add_argument("--s", type=float, default=0.0,
                    help="本次标定的靶纸表观边长 px（用于 k·s 拟合）")
    args = ap.parse_args()

    out = Path(args.out)

    # ---- 多点拟合
    if args.fit:
        pts = json.loads(Path(args.fit).read_text())
        d = fit_points(pts)
        write_calib(out, d["x"], d["y"], d["kx"], d["ky"], d["ref"],
                    method="fit", extra={"resid": round(d["resid"], 3),
                                         "n": d["n"]})
        if args.push:
            push_calib(out, args.port)
        return 0

    # ---- 手动坐标
    if args.x is not None and args.y is not None:
        write_calib(out, args.x, args.y, ref=args.s, method="manual")
        if args.push:
            push_calib(out, args.port)
        return 0

    # ---- 取图
    on = off = None
    if args.on and args.off:
        on, off = load_img(args.on), load_img(args.off)
    elif args.image:
        on = load_img(args.image)
    elif args.grab or (not args.on and not args.off and not args.image):
        esp_port = args.esp_port
        if args.pair and esp_port is None and not args.manual:
            for cand in ("/dev/ttyUSB0", "/dev/ttyACM1", "/dev/ttyUSB1"):
                if Path(cand).exists():
                    esp_port = cand
                    break
            if esp_port is None:
                _log("没找到 ESP32 串口 -> 改为手动开关激光（每步按回车）")
                args.manual = True
        if not args.pair:
            _log("单帧模式（激光常亮）：抓一帧后窗口里点一下光斑")
        on, off = grab_frames(args.port, pair=args.pair,
                              esp_port=esp_port, manual=args.manual)
        cv2.imwrite(str(ROOT / "live" / "calib_on.jpg"), on)
        _log("原图已存 live/calib_on.jpg")
        if off is not None:
            cv2.imwrite(str(ROOT / "live" / "calib_off.jpg"), off)
            _log("OFF 图已存 live/calib_off.jpg")
    else:
        ap.error("需要 --grab / --on+--off / --image / --x+--y / --fit 之一")

    # ---- 检测
    diff_mask = None
    if off is not None:
        cands, diff_mask = detect_diff(on, off)
        mode = "diff"
    else:
        cands, diff_mask = detect_single(on)
        mode = "single"
    if not cands:
        raise SystemExit("没检出候选；检查激光是否真的亮了/画面是否过曝")

    # 坐标从图片尺寸换算到输出坐标系
    ih, iw = on.shape[:2]
    for c in cands:
        c["x"], c["y"] = _scale_to_coord(c["x"], c["y"], iw, ih,
                                         args.coord_w, args.coord_h)
    best = cands[0]
    _log("候选 %d 个，最佳 (%.1f, %.1f) score=%.1f" %
         (len(cands), best["x"], best["y"], best["score"]))

    # ---- GUI 确认
    pt = (best["x"], best["y"])
    if not args.no_gui:
        try:
            view = _annotate(on, cands, 0)
            if diff_mask is not None:
                dm = cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR)
                h = min(view.shape[0], dm.shape[0])
                view = np.vstack([view, dm[:h]])
            picker = _Picker("laser calib: click the spot")
            got = picker.run(view, init_pt=(best["x"] * iw / args.coord_w,
                                            best["y"] * ih / args.coord_h))
            if picker.quit:
                _log("已取消")
                return 1
            if got is not None:
                pt = _scale_to_coord(got[0], got[1], iw, ih,
                                     args.coord_w, args.coord_h)
            if picker.rerun:
                _log("r 重跑：请重新执行本命令（不做半自动重试）")
                return 1
        except cv2.error as e:
            _log("无窗口环境（%s），自动接受最佳候选；需要人工确认请加显示环境" % e)

    d = write_calib(out, pt[0], pt[1], ref=args.s, method=mode)
    if args.push:
        push_calib(out, args.port)
    else:
        _log("下一步：python3 scripts/calib_laser.py --push 或手动 push 到 %s"
             % BOARD_CALIB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
