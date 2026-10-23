# -*- coding: utf-8 -*-
"""
export_onnx.py —— Ultralytics YOLOv8 权重导出 ONNX（K230 版）
============================================================

K230 的 CanMV `libs/YOLO` 期望「未做 NMS 的原始输出」：
    [1, 4+nc, 2100]   （输入 320x320 时 2100 = 40*40 + 20*20 + 10*10）
所以导出时必须关掉 NMS/end2end，输出才是 4+nc 通道。

用法：
    python3 tools/export_onnx.py --weights runs/detect/train/weights/best.pt \
        --imgsz 320 --out best.onnx

    # 自带权重（仅用于验证工具链）
    python3 tools/export_onnx.py --weights yolov8n.pt --out yolov8n_320.onnx
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
    ap = argparse.ArgumentParser(description="YOLOv8 -> ONNX（K230）")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nms", action="store_true", help="导出内置 NMS（实验）")
    args = ap.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[x] 需要 ultralytics：pip install ultralytics")
        return 2

    weights = Path(args.weights)
    if not weights.exists():
        print("[x] 找不到权重：%s" % weights)
        return 2
    model = YOLO(str(weights))
    print("[*] 模型：%s  类别数=%d" % (weights, len(model.names)))
    print("[*] 导出 ONNX imgsz=%d opset=%d …" % (args.imgsz, args.opset))
    kwargs = dict(format="onnx", imgsz=args.imgsz, opset=args.opset,
                  simplify=not args.no_simplify, dynamic=False,
                  nms=bool(args.nms), batch=1, device=args.device, half=False)
    try:
        path = model.export(**kwargs)
    except TypeError as exc:
        # 老版本没有 nms 参数
        print("[warn] 当前 ultralytics 不支持 nms=False（%s），重试" % exc)
        kwargs.pop("nms", None)
        path = model.export(**kwargs)
    path = Path(path)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.replace(target)
        except OSError:
            # 跨磁盘（比如权重在移动硬盘上）时 replace 会失败
            import shutil
            shutil.copy2(path, target)
            try:
                path.unlink()
            except OSError:
                pass
        path = target
    print("[✓] ONNX：%s（%.2f MB）" % (path, path.stat().st_size / 1048576.0))

    # 打印输出 shape，确认是 4+nc（没有 NMS）
    try:
        import onnx
        m = onnx.load(str(path))
        outs = m.graph.output
        print("[*] 输出张量：")
        for o in outs:
            dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
            print("    %s %s" % (o.name, dims))
        if outs:
            dims = [d.dim_value for d in outs[0].type.tensor_type.shape.dim]
            nc = len(model.names)
            is_seg = len(outs) >= 2 and len(dims) == 3
            ch = dims[1] if len(dims) == 3 else 0
            if is_seg:
                expect = 4 + nc + 32
                if ch and ch != expect:
                    print("[!] 警告：分割输出通道 %s != %d" % (ch, expect))
                else:
                    print("[✓] 分割模型输出正常：检测头 %s + mask 原型 %s"
                          % (dims, [d.dim_value for d in outs[1].type.tensor_type.shape.dim]))
            elif ch:
                rest = ch - 4 - nc
                if rest == 0:
                    print("[✓] 检测模型输出正常：%s（4+nc）" % dims)
                elif rest > 0 and rest % 3 == 0:
                    print("[✓] Pose 模型输出正常：%s（4+nc+3×%d 关键点）"
                          % (dims, rest // 3))
                else:
                    print("[!] 警告：输出通道 %s 不符合 4+nc / 4+nc+3k / 4+nc+32" % ch)
    except Exception as exc:
        print("[warn] 读取 ONNX 输出失败：%s" % exc)

    print("\n下一步：python3 tools/compile_kmodel.py --onnx %s --input-size %d "
          "--calib-dir captures" % (path, args.imgsz))
    return 0


if __name__ == "__main__":
    sys.exit(main())
