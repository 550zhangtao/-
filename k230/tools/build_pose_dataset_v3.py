#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_pose_dataset_v3.py —— 构建「小目标 + 强负样本」Pose 训练集

为什么要重建数据集
------------------
现场（2026-10-21）实测：部署的 poseB 对 35x52 px 的远处靶纸 **0/60 命中**，
同时把笔记本电脑 **60/60 误检**成靶纸；spot_v1 能检出 56/60，但笔电仍
误检 22/60。根因是训练集里：

1. 靶纸偏大：min-side 中位数 100 px，<60 px 的样本只占 7%，完全没有 <50 px；
2. 硬负样本太少：spot_v1 只继承了 v2_B 的 20 张空标注，丢掉了 farneg 的 80 张；
3. 没有「激光打在远处小靶上」的组合样本。

做法
----
* 以 v2_B + farneg_B 的 train 并集为底（内容哈希去重），保留全部 80 张负样本；
* 每张正样本生成：
    - 2 张叠加激光光斑（60% 压靶心 / 25% 靶纸其他位置 / 15% 靶外）
    - 1 张缩小到 0.30~0.65 倍（模拟远处小靶）
    - 1 张缩小 + 靶心叠光斑（远处小靶被激光压住 —— 现场最难的组合）
* 负样本：原始 80 张 + 每张随机裁 2 块（学不同尺度）+ 现场帧里
  避开靶纸的随机裁块（笔电/键盘/纸箱等硬负样本）；
* 现场连拍帧用 label_scene_frames.py 生成的中值标注，按 1/6、2/6、3/6
  划分 train/val/test，互不重叠。

产物
----
out/images|labels/{train,val,test}、data.yaml、stats.json、samples.jpg

用法
----
    python3 tools/build_pose_dataset_v3.py \
        --sources ../recovered/k230_target_project/dataset_pose_v2_B \
                  ../recovered/k230_target_project/dataset_pose_farneg_B \
        --val-src ../recovered/k230_target_project/dataset_pose_farneg_B \
        --test-src ../recovered/k230_target_project/dataset_pose_farneg_B \
        --scene-frames /tmp/scene_frames --scene-labels /tmp/scene_ds/scene_labels.json \
        --out ../recovered/dataset_pose_v3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

KPT = 5


# ---------------------------------------------------------------- 标签
def read_label(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    lines = [l.split() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return None
    t = lines[0]
    if len(t) < 5 + KPT * 3:
        return None
    box = np.array([float(v) for v in t[1:5]], np.float32)
    pts = np.array([[float(t[5 + i * 3]), float(t[6 + i * 3]), float(t[7 + i * 3])]
                    for i in range(KPT)], np.float32)
    return box, pts


def write_label(path: Path, box: np.ndarray, pts: np.ndarray, w: int, h: int) -> None:
    """box 为归一化 cx,cy,w,h；pts 为归一化 x,y,v。"""
    vals = [0, *[float(v) for v in box]]
    for x, y, v in pts:
        vals += [float(x), float(y), float(v)]
    path.write_text(" ".join("%.6f" % v for v in vals) + "\n", encoding="utf-8")


def box_to_px(box_n: np.ndarray, w: int, h: int) -> np.ndarray:
    cx, cy, bw, bh = box_n
    return np.array([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h], np.float32)


def px_to_box(box_px: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box_px
    return np.array([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                     (x2 - x1) / w, (y2 - y1) / h], np.float32)


def xform(box_px: np.ndarray, kps_px: np.ndarray, s: float,
          ox: float, oy: float) -> Tuple[np.ndarray, np.ndarray]:
    box = box_px.copy()
    box[[0, 2]] = box[[0, 2]] * s + ox
    box[[1, 3]] = box[[1, 3]] * s + oy
    kp = kps_px.copy()
    kp[:, 0] = kp[:, 0] * s + ox
    kp[:, 1] = kp[:, 1] * s + oy
    return box, kp


# ---------------------------------------------------------------- 增强
def add_laser(img: np.ndarray, x: float, y: float, rad: float, rng: random.Random) -> None:
    """与 augment_spot.py 完全一致的激光光斑合成（BGR）。"""
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < w and 0 <= y < h):
        return
    rings = [
        (rad * 2.4, (40, 25, 130)),
        (rad * 1.7, (55, 40, 190)),
        (rad * 1.15, (75, 55, 235)),
        (rad * 0.75, (110, 85, 250)),
    ]
    for r, col in rings:
        cv2.circle(img, (x, y), int(max(1, r)), col, -1, cv2.LINE_AA)
    core = max(1, int(round(rad * rng.uniform(0.30, 0.60))))
    cv2.circle(img, (x, y), core, (250, 252, 255), -1, cv2.LINE_AA)
    k = int(max(3, rad * 1.2))
    if k % 2 == 0:
        k += 1
    x0, y0 = max(0, x - k), max(0, y - k)
    x1, y1 = min(w, x + k), min(h, y + k)
    if x1 - x0 > 2 and y1 - y0 > 2:
        img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (3, 3), 0)


def laser_at_target(img: np.ndarray, box_px: np.ndarray, kps: np.ndarray,
                    rng: random.Random, mode: str = "center") -> None:
    """在靶纸/靶心附近叠一个按距离缩放的光斑（原地修改）。"""
    x1, y1, x2, y2 = box_px
    bw, bh = x2 - x1, y2 - y1
    if mode == "center":
        lx = kps[4, 0] + rng.uniform(-bw * 0.07, bw * 0.07)
        ly = kps[4, 1] + rng.uniform(-bh * 0.07, bh * 0.07)
    elif mode == "target":
        lx = rng.uniform(x1, x2)
        ly = rng.uniform(y1, y2)
    else:
        lx = rng.uniform(0, img.shape[1] - 1)
        ly = rng.uniform(0, img.shape[0] - 1)
    ref = max(24.0, min(bw, bh))
    rad = max(1.5, min(14.0, rng.uniform(0.020, 0.075) * ref))
    add_laser(img, lx, ly, rad, rng)


def zoom_out(img: np.ndarray, s: float, rng: random.Random) -> Tuple[np.ndarray, float, int, int]:
    """整图缩小后贴回 640x480 画布（边缘 replicate），返回画布/比例/偏移。"""
    h, w = img.shape[:2]
    nw, nh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ox = rng.randint(0, max(0, w - nw))
    oy = rng.randint(0, max(0, h - nh))
    canvas = cv2.copyMakeBorder(small, oy, h - nh - oy, ox, w - nw - ox,
                                cv2.BORDER_REPLICATE)
    return canvas, s, ox, oy


# ---------------------------------------------------------------- 数据收集
def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(dir_: Path) -> List[Dict]:
    out, seen = [], set()
    img_dir, lab_dir = dir_ / "images" / "train", dir_ / "labels" / "train"
    if not img_dir.is_dir():
        return out
    for p in sorted(img_dir.glob("*.jpg")):
        h = md5(p)
        if h in seen:
            continue
        seen.add(h)
        out.append({"src": p, "label": lab_dir / (p.stem + ".txt"), "hash": h})
    return out


def collect_split(dir_: Path, split: str) -> List[Dict]:
    out, seen = [], set()
    img_dir, lab_dir = dir_ / "images" / split, dir_ / "labels" / split
    if not img_dir.is_dir():
        return out
    for p in sorted(img_dir.glob("*.jpg")):
        h = md5(p)
        if h in seen:
            continue
        seen.add(h)
        out.append({"src": p, "label": lab_dir / (p.stem + ".txt"), "hash": h})
    return out


def random_crops(img: np.ndarray, exclude: Optional[np.ndarray], n: int,
                 rng: random.Random) -> List[np.ndarray]:
    """在图上随机裁块，避免与 exclude（x1,y1,x2,y2，已外扩）相交。"""
    h, w = img.shape[:2]
    out = []
    for _ in range(n):
        for _try in range(30):
            cw = rng.randint(220, 460)
            ch = rng.randint(220, 460)
            if cw >= w or ch >= h:
                continue
            x = rng.randint(0, w - cw)
            y = rng.randint(0, h - ch)
            if exclude is not None:
                ex1, ey1, ex2, ey2 = exclude
                if not (x + cw < ex1 or x > ex2 or y + ch < ey1 or y > ey2):
                    continue
            out.append(img[y:y + ch, x:x + cw].copy())
            break
    return out


def draw_samples(items: List[Dict], out_path: Path, n: int = 12) -> None:
    """抽 12 张画框+关键点，拼 4x3 图供人工核对。"""
    rng = random.Random(0)
    picks = rng.sample(items, min(n, len(items)))
    tiles = []
    for it in picks:
        img = cv2.imread(str(it["path"]))
        lab = read_label(it["label"])
        if img is None:
            continue
        if lab:
            box, pts = lab
            h, w = img.shape[:2]
            b = box_to_px(box, w, h)
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            for i in range(KPT):
                x, y = pts[i, 0] * w, pts[i, 1] * h
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255) if i < 4 else (255, 0, 255), -1)
        img = cv2.resize(img, (320, 240))
        cv2.putText(img, it["path"].name[-34:], (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    sheet = np.vstack(rows) if len(rows) > 1 else rows[0]
    cv2.imwrite(str(out_path), sheet)
    print("[✓] 抽检图 -> %s" % out_path)


# ---------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="构建 v3 Pose 数据集（小目标+硬负样本）")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="含 images/train 与 labels/train 的数据集目录（可多个，自动去重）")
    ap.add_argument("--val-src", required=True)
    ap.add_argument("--test-src", required=True)
    ap.add_argument("--scene-frames", default=None)
    ap.add_argument("--scene-labels", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spot-copies", type=int, default=2)
    ap.add_argument("--zoom-copies", type=int, default=2)
    ap.add_argument("--neg-crops", type=int, default=2, help="每张负样本再随机裁几块")
    ap.add_argument("--scene-neg-crops", type=int, default=2, help="每张现场帧裁几块硬负样本")
    ap.add_argument("--scene-train", type=int, default=10,
                    help="现场帧中用于训练的帧数（其余按 2:3 给 val/test）")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    for sub in ("images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    train_items: List[Dict] = []
    seen = set()
    for s in args.sources:
        for it in collect(Path(s)):
            if it["hash"] in seen:
                continue
            seen.add(it["hash"])
            train_items.append(it)
    pos = [it for it in train_items if read_label(it["label"]) is not None]
    neg = [it for it in train_items if read_label(it["label"]) is None]
    print("[*] 训练底图：%d（正 %d / 负 %d，已按内容去重）" % (len(train_items), len(pos), len(neg)))

    stats = {"pos_src": len(pos), "neg_src": len(neg), "spot": 0, "zoom": 0,
             "zoom_spot": 0, "neg_crops": 0, "scene_train": 0, "scene_val": 0,
             "scene_test": 0, "scene_neg": 0}

    def add(path_img: Path, label_path: Path) -> None:
        shutil.copy2(path_img, out / "images" / "train" / path_img.name)
        if label_path.exists():
            shutil.copy2(label_path, out / "labels" / "train" / label_path.name)
        else:
            (out / "labels" / "train" / (path_img.stem + ".txt")).write_text("")

    # ---- 正样本：原图 + 光斑 + 缩小 + 缩小+光斑
    for it in pos:
        img = cv2.imread(str(it["src"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        lab = read_label(it["label"])
        if lab is None:
            continue
        box_n, pts_n = lab
        box_px = box_to_px(box_n, w, h)
        kps = pts_n.copy()
        kps[:, 0] *= w
        kps[:, 1] *= h
        add(it["src"], it["label"])

        for k in range(args.spot_copies):
            canvas = img.copy()
            mode = "center" if rng.random() < 0.60 else ("target" if rng.random() < 0.625 else "off")
            laser_at_target(canvas, box_px, kps, rng, mode)
            name = "%s_spot%d" % (it["src"].stem, k)
            p = out / "images" / "train" / (name + ".jpg")
            cv2.imwrite(str(p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
            write_label(out / "labels" / "train" / (name + ".txt"), box_n, pts_n, w, h)
            stats["spot"] += 1

        for k in range(args.zoom_copies):
            s = rng.uniform(0.30, 0.65)
            canvas, s, ox, oy = zoom_out(img, s, rng)
            b2, k2 = xform(box_px, kps, s, ox, oy)
            with_spot = (k == args.zoom_copies - 1)  # 最后一张加光斑
            if with_spot:
                laser_at_target(canvas, b2, k2, rng, "center")
            name = "%s_zoom%d" % (it["src"].stem, k)
            p = out / "images" / "train" / (name + ".jpg")
            cv2.imwrite(str(p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
            write_label(out / "labels" / "train" / (name + ".txt"),
                        px_to_box(b2, w, h), np.column_stack([k2[:, :2] / [w, h], k2[:, 2:]]), w, h)
            stats["zoom" if not with_spot else "zoom_spot"] += 1

    # ---- 负样本：原图 + 随机裁块
    for it in neg:
        add(it["src"], it["label"])
        img = cv2.imread(str(it["src"]))
        if img is None:
            continue
        for k, crop in enumerate(random_crops(img, None, args.neg_crops, rng)):
            name = "%s_negcrop%d" % (it["src"].stem, k)
            p = out / "images" / "train" / (name + ".jpg")
            cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / "train" / (name + ".txt")).write_text("")
            stats["neg_crops"] += 1

    # ---- 现场帧：正样本按 1/6 : 2/6 : 3/6 划分；另裁硬负样本
    if args.scene_frames and args.scene_labels:
        man = json.loads(Path(args.scene_labels).read_text(encoding="utf-8"))
        box_n = np.array([(man["box_px"][0] + man["box_px"][2]) / 2 / man["size"][0],
                          (man["box_px"][1] + man["box_px"][3]) / 2 / man["size"][1],
                          (man["box_px"][2] - man["box_px"][0]) / man["size"][0],
                          (man["box_px"][3] - man["box_px"][1]) / man["size"][1]], np.float32)
        pts_n = np.array([[p[0] / man["size"][0], p[1] / man["size"][1], 2]
                          for p in man["kpts_px"]], np.float32)
        frames = sorted(Path(args.scene_frames).glob("*.jpg"))
        n_tr = min(args.scene_train, max(1, len(frames) // 6))
        for i, p in enumerate(frames):
            if i < n_tr:
                split, cnt = "train", "scene_train"
            elif i % 3 == 0:
                split, cnt = "val", "scene_val"
            else:
                split, cnt = "test", "scene_test"
            shutil.copy2(p, out / "images" / split / p.name)
            write_label(out / "labels" / split / (p.stem + ".txt"), box_n, pts_n,
                        man["size"][0], man["size"][1])
            stats[cnt] += 1
            if split == "train" and args.scene_neg_crops:
                img = cv2.imread(str(p))
                h, w = img.shape[:2]
                ex = np.array(man["box_px"], np.float32)
                ex = np.array([ex[0] - 12, ex[1] - 12, ex[2] + 12, ex[3] + 12])
                for k, crop in enumerate(random_crops(img, ex, args.scene_neg_crops, rng)):
                    name = "%s_hneg%d" % (p.stem, k)
                    cv2.imwrite(str(out / "images" / "train" / (name + ".jpg")), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    (out / "labels" / "train" / (name + ".txt")).write_text("")
                    stats["scene_neg"] += 1

    # ---- val / test：只放原始样本（不做合成增强）
    for split, src in (("val", args.val_src), ("test", args.test_src)):
        for it in collect_split(Path(src), split):
            shutil.copy2(it["src"], out / "images" / split / it["src"].name)
            if it["label"].exists():
                shutil.copy2(it["label"], out / "labels" / split / it["label"].name)
            else:
                (out / "labels" / split / (it["src"].stem + ".txt")).write_text("")

    (out / "data.yaml").write_text(
        "path: %s\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['target']\n"
        "kpt_shape: [5, 3]\n"
        "flip_idx: [1, 0, 3, 2, 4]\n" % out.resolve(), encoding="utf-8")

    counts = {}
    for split in ("train", "val", "test"):
        imgs = list((out / "images" / split).glob("*.jpg"))
        labs = list((out / "labels" / split).glob("*.txt"))
        empty = sum(1 for l in labs if not l.read_text(encoding="utf-8").strip())
        counts[split] = {"images": len(imgs), "labels": len(labs),
                         "pos": len(imgs) - empty, "neg": empty}
    stats["counts"] = counts
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print("[✓] 数据集 -> %s" % out)

    items = [{"path": p, "label": out / "labels" / "train" / (p.stem + ".txt")}
             for p in (out / "images" / "train").glob("*.jpg")]
    draw_samples(items, out / "samples.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
