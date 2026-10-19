# -*- coding: utf-8 -*-
"""
dataset_from_captures.py —— 把 K230 抓的图整理成 YOLO 数据集
============================================================

输入：
    captures/       K230 抓的 jpg/png（grab/live 产生）
    labels/         同名的 YOLO txt（可选；没有就只建目录结构）

输出：
    dataset/
        images/train, images/val
        labels/train, labels/val
        data.yaml

用法：
    python3 tools/dataset_from_captures.py --images captures --labels labels \
        --out dataset --names target --val-ratio 0.2

    # 没有人工标注时，先用 Hough 圆检测生成粗标：
    python3 tools/autolabel_target.py --images captures --out labels --preview preview
    # 然后人工复核 txt，再跑本脚本
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_names(text: str) -> list:
    text = text.strip()
    if text.startswith("["):
        return [str(x) for x in json.loads(text)]
    return [x.strip() for x in text.split(",") if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="整理 YOLO 数据集")
    ap.add_argument("--images", required=True, help="图片目录")
    ap.add_argument("--labels", default=None, help="YOLO txt 目录（可缺省）")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--names", default="target", help="逗号分隔类别名")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--symlink", action="store_true", help="用软链接代替复制")
    ap.add_argument("--neg-only-from", default=None,
                    help="只有这些目录（逗号分隔，如 neg）的空标签算负样本；"
                         "其它空标签视为“未标注”，直接跳过")
    args = ap.parse_args(argv)

    img_root = Path(args.images)
    if not img_root.is_dir():
        print("[x] 图片目录不存在：%s" % img_root)
        return 2
    images = [p for p in sorted(img_root.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not images:
        print("[x] %s 里没有图片" % img_root)
        return 2

    label_root = Path(args.labels) if args.labels else None
    neg_only = ({x.strip() for x in args.neg_only_from.split(",") if x.strip()}
                if args.neg_only_from else set())
    names = parse_names(args.names)
    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rng.shuffle(images)
    n_val = max(1, int(round(len(images) * args.val_ratio))) if len(images) > 1 else 0
    val = set(images[:n_val])
    counts = {"train": 0, "val": 0, "labeled": 0, "empty": 0}

    skipped_unlabeled = 0
    for p in images:
        split = "val" if p in val else "train"
        # 不同阶段可能有同名 frame_000000.jpg，加阶段前缀避免覆盖
        stage = p.parent.name if p.parent != img_root else ""
        fname = (stage + "_" + p.stem) if stage else p.stem
        dst_img = out / "images" / split / (fname + p.suffix.lower())
        if args.symlink:
            try:
                dst_img.symlink_to(p.resolve())
            except OSError:
                shutil.copy2(p, dst_img)
        else:
            shutil.copy2(p, dst_img)
        dst_lbl = out / "labels" / split / (fname + ".txt")
        src_lbl = None
        if label_root:
            # 1) 同阶段子目录优先：captures/pos_far/x.jpg -> labels/pos_far/x.txt
            same = label_root / stage / (p.stem + ".txt")
            if same.exists():
                src_lbl = same
            else:
                cand = label_root / (p.stem + ".txt")
                if cand.exists():
                    src_lbl = cand
                else:
                    hits = list(label_root.rglob(p.stem + ".txt"))
                    if hits:
                        src_lbl = hits[0]
        if src_lbl and src_lbl.exists():
            txt = src_lbl.read_text(encoding="utf-8").strip()
            if txt:
                shutil.copy2(src_lbl, dst_lbl)
                counts["labeled"] += 1
            elif neg_only and p.parent.name not in neg_only:
                # 未标注（自动标注被拒）的帧，不当负样本，直接跳过
                dst_img.unlink(missing_ok=True)
                skipped_unlabeled += 1
                continue
            else:
                dst_lbl.write_text("", encoding="utf-8")
                counts["empty"] += 1
        else:
            dst_lbl.write_text("", encoding="utf-8")
            counts["empty"] += 1
        counts[split] += 1

    yaml = out / "data.yaml"
    lines = [
        "path: %s" % out.resolve(),
        "train: images/train",
        "val: images/val",
        "nc: %d" % len(names),
        "names: [%s]" % ", ".join("'%s'" % n for n in names),
    ]
    yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[✓] 数据集：%s" % out.resolve())
    print("    train=%d val=%d  有标注=%d 负样本=%d  跳过未标注=%d  类别=%s" %
          (counts["train"], counts["val"], counts["labeled"],
           counts["empty"], skipped_unlabeled, names))
    if counts["empty"]:
        print("[!] 有 %d 张图没有标注（空 txt 会被当成负样本）。"
              "建议先跑 tools/autolabel_target.py 再人工复核。" % counts["empty"])
    print("\n下一步：python3 tools/train_target.py --data %s" % yaml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
