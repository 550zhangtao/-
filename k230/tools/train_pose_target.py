# -*- coding: utf-8 -*-
"""
train_pose_target.py —— 训练 5 关键点 YOLOv8n-Pose（靶纸角点 + 靶心）
====================================================================

关键点顺序（固定）：
    1 TL 左上角       2 TR 右上角
    3 BR 右下角       4 BL 左下角
    5 白色 10 号靶心中心

设计思路：
    检测框只负责给出“靶纸特征区域”，四角和靶心由关键点分支回归。
    框不需要卡得极准，只要完整、合理地覆盖靶纸即可。

数据格式（YOLO-Pose，每行）：
    cls cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4  x5 y5 v5
坐标归一化 0~1；v=2 可见，v=1 被遮挡/推断，v=0 未标。

data.yaml：
    kpt_shape: [5, 3]
    flip_idx: [1, 0, 3, 2, 4]

用法：
    python3 tools/train_pose_target.py --data dataset_pose/data.yaml \
        --model yolov8n.pt --epochs 100 --imgsz 320 --batch 8 --device cpu

导出后输出应为 [1, 20, 2100]（4+nc+3*5，nc=1，320 输入）。
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
import sys
from pathlib import Path

def check_pose_dataset(yaml_path: Path, kpt: int) -> bool:
    root = yaml_path.parent
    need = 5 + kpt * 3
    ok = True
    for split in ("train", "val"):
        ldir = root / "labels" / split
        if not ldir.is_dir():
            continue
        good = bad = 0
        for f in ldir.glob("*.txt"):
            for line in f.read_text(encoding="utf-8").splitlines():
                t = line.split()
                if len(t) >= need:
                    good += 1
                elif t:
                    bad += 1
        print("  %-5s 合法关键点行 %5d  非法 %5d" % (split, good, bad))
        if bad:
            ok = False
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练 5 关键点 YOLOv8n-Pose")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt",
                    help="预训练权重（检测/姿态均可）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--project", default="runs_target")
    ap.add_argument("--name", default="pose")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--freeze-kpt", action="store_true",
                    help="冻结关键点分支 cv4，只微调检测/分类")
    ap.add_argument("--kpt", type=int, choices=[4, 5], default=5,
                   help="A=4 角点，B=4 角点+靶心")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)

    data = Path(args.data)
    if not data.exists():
        print("[x] 找不到 %s" % data)
        return 2
    text = data.read_text(encoding="utf-8")
    if "kpt_shape" not in text:
        print("[!] data.yaml 缺少 kpt_shape: [5, 3]，训练大概率失败")
        return 3

    if not args.no_check:
        print("[*] 检查关键点标注（每行需要 %d 个字段）…" % (5 + args.kpt * 3))
        if not check_pose_dataset(data, args.kpt):
            print("\n[x] 有关键点行格式不对。")
            print("    格式：cls cx cy w h  x1 y1 v1 … x%d y%d v%d" % (args.kpt, args.kpt, args.kpt))
            print("    顺序：TL, TR, BR, BL" + (", 靶心" if args.kpt == 5 else ""))
            print("    详见 docs/POSE_ANNOTATION.md")
            return 3

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[x] 需要 ultralytics")
        return 2

    model = YOLO("yolov8n-pose.yaml")
    if args.model and Path(args.model).exists():
        model = model.load(args.model)
        print("[*] 从 %s 迁移初始化" % args.model)
    if args.freeze_kpt:
        n = 0
        for name, p in model.model.named_parameters():
            if "cv4" in name:
                p.requires_grad = False
                n += 1
        print("[*] 已冻结关键点分支参数 %d 个（cv4）" % n)

    print("[*] 训练 pose imgsz=%d epochs=%d batch=%d kpt=%d" %
          (args.imgsz, args.epochs, args.batch, args.kpt))
    model.train(
        data=str(data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.project).resolve()),
        name=args.name,
        patience=args.patience,
        lr0=args.lr0,
        freeze=args.freeze,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        cache=False,
        # 光照/姿态/遮挡增强
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,
        degrees=20.0, translate=0.15, scale=0.7, shear=5.0,
        perspective=0.001,
        fliplr=0.5,
        # ★ Pose 不要开 flipud：data.yaml 的 flip_idx 只描述左右镜像
        #   （TL/TR/BR/BL），垂直翻转后 ultralytics 仍套用这个映射，
        #   关键点语义会被打乱（2026-10-21 查明是精度退化原因之一）。
        flipud=0.0,
        mosaic=1.0, close_mosaic=10,
        # ★ 不要开 erasing：随机擦除会盖掉靶心/角点，逼模型凭空回归。
        erasing=0.0,
        plots=True,
    )
    best = Path(args.project).resolve() / args.name / "weights" / "best.pt"
    print("\n[✓] 训练完成：%s" % best)
    print("下一步：python3 tools/export_onnx.py --weights %s --imgsz %d --out pose.onnx"
          % (best, args.imgsz))
    return 0


if __name__ == "__main__":
    sys.exit(main())
