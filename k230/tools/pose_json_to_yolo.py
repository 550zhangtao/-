# -*- coding: utf-8 -*-
"""
pose_json_to_yolo.py —— 标注工具导出的 JSON -> YOLO-Pose A/B 数据集
=================================================================

输入：pose_label.html 导出的 pose_labels.json
输出：
    dataset_pose_A/   4 角点（TL,TR,BR,BL）
    dataset_pose_B/   4 角点 + 靶心（第 5 点）

两个版本使用**同一批图、同一批四角标注**，只有第 5 点有无的差别，
保证 A/B 训练条件可比。

用法：
    python3 tools/pose_json_to_yolo.py --json ../pose_labels.json \
        --captures ../k230_target_project/captures \
        --out-root ../k230_target_project

    # 只生成 B，或自定义划分
    python3 tools/pose_json_to_yolo.py --json ../pose_labels.json \
        --captures ../k230_target_project/captures \
        --out-root ../k230_target_project --versions B \
        --test-every 10 --val-every 10
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
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ORDER = ["TL", "TR", "BR", "BL", "bull"]


def load_json(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "images" not in data:
        raise SystemExit("JSON 里没有 images 字段")
    return data


def frame_no(key: str) -> int:
    name = Path(key).stem
    digits = "".join(ch if ch.isdigit() else " " for ch in name).split()
    return int(digits[-1]) if digits else 0


def split_of(key: str, test_every: int, val_every: int, seed: int,
             split_map: Dict[str, str] = None) -> str:
    """按帧号取模划分，避免相邻帧同时进训练/测试造成泄漏。"""
    if split_map is not None:
        return split_map.get(key, "train")
    n = frame_no(key)
    if test_every and n % test_every == 0:
        return "test"
    if val_every and n % val_every == 0:
        return "val"
    return "train"


def points_to_line(pts: List[List[float]], w: int, h: int, kpt: int) -> str:
    """pts: [[x,y,v], ...] 像素坐标；返回归一化 YOLO-Pose 行。"""
    xs = [float(p[0]) for p in pts[:4]]
    ys = [float(p[1]) for p in pts[:4]]
    x1 = max(0.0, min(xs)); x2 = min(float(w), max(xs))
    y1 = max(0.0, min(ys)); y2 = min(float(h), max(ys))
    bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    line = "0 %.6f %.6f %.6f %.6f" % (cx / w, cy / h, bw / w, bh / h)
    for i in range(kpt):
        p = pts[i]
        px = max(0.0, min(1.0, float(p[0]) / w))
        py = max(0.0, min(1.0, float(p[1]) / h))
        v = int(p[2]) if len(p) > 2 else 2
        line += " %.6f %.6f %d" % (px, py, v)
    return line


def make_version(root: Path, data: Dict, captures: Path, version: str,
                 kpt: int, test_every: int, val_every: int, seed: int,
                 split_map: Dict[str, str] = None, tag: str = "") -> Dict:
    name = "dataset_pose%s_%s" % (("_" + tag) if tag else "", version)
    base = root / name
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (base / sub / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    vcount = [0, 0, 0]
    for key, rec in sorted(data["images"].items()):
        if not rec.get("done"):
            continue
        pts = rec.get("points") or []
        negative = bool(rec.get("negative"))
        if not negative and len(pts) < kpt:
            continue
        if not negative and kpt == 5 and len(pts) < 5:
            continue
        w, h = int(rec.get("w") or 640), int(rec.get("h") or 480)
        split = split_of(key, test_every, val_every, seed, split_map)
        src = captures / key
        if not src.exists():
            # 兼容 captures/ 前缀
            src = captures / key.split("/", 1)[-1]
        if not src.exists():
            print("  [skip] 找不到图片 %s" % src)
            continue
        kp = Path(key)
        stem = (kp.parent.name + "_" + kp.stem) if kp.parent.name else kp.stem
        dst = base / "images" / split / (stem + src.suffix)
        if not dst.exists():
            shutil.copy2(src, dst)
        if negative:
            (base / "labels" / split / (stem + ".txt")).write_text("", encoding="utf-8")
        else:
            line = points_to_line(pts, w, h, kpt)
            (base / "labels" / split / (stem + ".txt")).write_text(line + "\n", encoding="utf-8")
            for p in pts[:kpt]:
                if len(p) > 2:
                    vcount[int(p[2])] += 1
        counts[split] += 1

    # Ultralytics 要求 val 非空；小数据集时从 train 挪 1~2 张
    for split in ("val", "test"):
        if not list((base / "images" / split).glob("*")):
            tr = sorted((base / "images" / "train").glob("*"))
            if len(tr) > 1:
                src_img = tr[0]
                src_lbl = base / "labels" / "train" / (src_img.stem + ".txt")
                dst_img = base / "images" / split / src_img.name
                dst_lbl = base / "labels" / split / src_lbl.name
                src_img.rename(dst_img)
                if src_lbl.exists():
                    src_lbl.rename(dst_lbl)
                counts[split] += 1
                counts["train"] -= 1

    yaml = base / "data.yaml"
    flip = "[1, 0, 3, 2, 4]" if kpt == 5 else "[1, 0, 3, 2]"
    yaml.write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames: ['target']\nkpt_shape: [%d, 3]\nflip_idx: %s\n"
        % (base.resolve(), kpt, flip), encoding="utf-8")
    print("[✓] %s：train %d / val %d / test %d，kpt=%d" %
          (name, counts["train"], counts["val"], counts["test"], kpt))
    print("    关键点可见性分布 v0=%d v1=%d v2=%d" % tuple(vcount))
    return {"name": name, "counts": counts, "v": vcount}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pose_labels.json -> YOLO-Pose A/B")
    ap.add_argument("--json", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--versions", default="AB", help="A=4角点，B=5关键点，如 AB/B")
    ap.add_argument("--test-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="数据集后缀，如 v2 -> dataset_pose_v2_A")
    ap.add_argument("--split-file", default=None,
                    help='JSON: {"test":[key...],"val":[key...]}（可选，覆盖按帧号划分）')
    args = ap.parse_args(argv)

    data = load_json(Path(args.json))
    caps = Path(args.captures)
    root = Path(args.out_root)
    versions = [v for v in args.versions.upper() if v in "AB"]
    if not versions:
        print("[x] --versions 需要 A 和/或 B")
        return 2
    total = sum(1 for r in data["images"].values() if r.get("done"))
    print("[*] 已完成标注 %d 张，生成版本 %s" % (total, ",".join(versions)))
    if total == 0:
        print("[x] 没有 done=true 的标注；在工具里点“完成并下一张”后再导出")
        return 3

    split_map = None
    if args.split_file:
        sm = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        split_map = {}
        for k in sm.get("test", []):
            split_map[k] = "test"
        for k in sm.get("val", []):
            split_map[k] = "val"
        print("[*] 使用 split-file：test %d / val %d" %
              (len(sm.get("test", [])), len(sm.get("val", []))))

    for v in versions:
        make_version(root, data, caps, v, 5 if v == "B" else 4,
                     args.test_every, args.val_every, args.seed, split_map, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
