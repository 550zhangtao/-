#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_pose_v3.py —— Pose 小目标微调（现场退化修复版）

和 `train_pose_target.py` 的区别（也是这次退化的两个训练侧根因）：

1. **flipud=0**：旧配方用 `flipud=0.5` 上下翻转。Pose 的关键点是有语义
   顺序的（TL/TR/BR/BL + 靶心），上下翻转后标签不会重排，等于给模型
   灌了一半错误标注；左右翻转有 flip_idx 保护，上下没有。
2. **erasing=0**：旧配方 `erasing=0.25` 会随机擦掉 1/4 区域，经常正好
   擦掉靶心关键点，迫使模型在"靶心消失"的图上回归靶心。

其余是面向小目标/强负样本的常规配方：
mosaic（最后 8 轮关闭）、适度 scale/translate/degrees、HSV、AdamW + cosine。

用法：
    python3 tools/train_pose_v3.py \
        --data ../recovered/dataset_pose_v3/data.yaml \
        --weights models/pose_v3.pt \
        --epochs 40 --imgsz 320 --batch 16 --device cpu \
        --project ../runs --name pose_v3
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pose 小目标微调")
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True, help="起点权重（pose .pt）")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--freeze", type=int, default=0,
                    help="冻结前 N 层（0=全量微调；小数据集可先 10）")
    ap.add_argument("--project", default="../runs")
    ap.add_argument("--name", default="pose_v3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    data = Path(args.data)
    if not data.exists():
        print("[x] 找不到 %s" % data)
        return 2
    yml = data.read_text(encoding="utf-8")
    if "kpt_shape" not in yml or "flip_idx" not in yml:
        print("[x] data.yaml 缺少 kpt_shape / flip_idx（Pose 必需）")
        return 3

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[x] 需要 ultralytics")
        return 2

    weights = Path(args.weights)
    if not weights.exists():
        print("[x] 找不到权重 %s" % weights)
        return 2
    model = YOLO(str(weights))
    print("[*] 起点 %s  task=%s  names=%s" % (weights, model.task, model.names))

    model.train(
        data=str(data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        freeze=args.freeze,
        cache=False if args.no_cache else "ram",
        # ---- 增强：全部对 Pose 安全 ----
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,
        degrees=12.0,
        translate=0.12,
        scale=0.55,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,          # ★ 关键：Pose 不能上下翻转
        mosaic=1.0,
        close_mosaic=8,
        erasing=0.0,         # ★ 关键：不能擦掉关键点
        plots=True,
        val=True,
        save=True,
        verbose=True,
    )
    best = Path(args.project).resolve() / args.name / "weights" / "best.pt"
    print("\n[✓] 训练完成：%s" % best)
    bench = Path(__file__).resolve().parent / "bench_pose.py"
    print("下一步：")
    print("  python3 %s --weights %s %s --data %s --confs 0.1 0.25 0.4 --out eval_v3.json"
          % (bench, weights, best, data.parent))
    print("  python3 tools/export_onnx.py --weights %s --imgsz %d --out pose_v3.onnx"
          % (best, args.imgsz))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
