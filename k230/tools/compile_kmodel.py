# -*- coding: utf-8 -*-
"""
compile_kmodel.py —— YOLOv8 ONNX -> K230 kmodel（nncase + KPU）
==============================================================

按嘉楠官方 K230 文档的 YOLOv8 示例配置：

    target=k230, preprocess=True, input_type=uint8,
    input_range=[0,1], mean=[0,0,0], std=[1,1,1], input_layout=NCHW
    PTQ: samples_count=10, calibrate=Kld/NoClip, quant=uint8

依赖：
    pip install nncase nncase-kpu onnx onnxsim onnxruntime numpy opencv-python
    .NET 7 runtime（DOTNET_ROOT，例如 ~/.dotnet）

用法：
    python3 tools/compile_kmodel.py --onnx best.onnx --out best.kmodel \
        --input-size 320 --calib-dir captures --samples 20

    # 校准集不够时会重复使用；完全不提供则用合成灰阶/噪声图（不推荐）
    python3 tools/compile_kmodel.py --onnx best.onnx --out best.kmodel --allow-synthetic-calib
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
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _prepare_dotnet() -> str:
    """让 nncase 找得到 .NET 7 的 hostfxr。"""
    candidates = [
        os.environ.get("DOTNET_ROOT"),
        os.path.expanduser("~/.dotnet"),
        "/usr/share/dotnet",
        "/usr/lib/dotnet",
        r"C:\Program Files\dotnet",
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "host")):
            os.environ["DOTNET_ROOT"] = c
            break
    # nncase-kpu 的算子模块（pip 安装后在 nncase/modules）。
    # 必须在 import nncase 之前设置，否则 PluginLoader 已经跑完了。
    try:
        import importlib.util
        spec = importlib.util.find_spec("nncase")
        if spec and spec.origin:
            mod = Path(spec.origin).resolve().parent / "modules"
            if mod.is_dir():
                os.environ.setdefault("NNCASE_PLUGIN_PATH", str(mod))
    except Exception:
        pass
    return os.environ.get("DOTNET_ROOT", "")


def parse_size(text: str) -> Tuple[int, int]:
    text = str(text).lower().replace(" ", "")
    if "x" in text:
        w, h = text.split("x", 1)
        return int(w), int(h)
    v = int(text)
    return v, v


def align32(v: int) -> int:
    return int(math.ceil(v / 32.0)) * 32


def load_calib_images(calib_dir: str, width: int, height: int,
                      samples: int, allow_synthetic: bool
                      ) -> Tuple[List[np.ndarray], str]:
    """返回 [N,1,3,H,W] uint8 校准数据。"""
    import cv2

    paths: List[Path] = []
    if calib_dir:
        root = Path(calib_dir)
        if root.is_file():
            paths = [root]
        elif root.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                paths.extend(sorted(root.rglob(ext)))
    if not paths:
        if not allow_synthetic:
            raise SystemExit(
                "校准集为空：请用 --calib-dir 指定一个图片目录（建议 20~100 张实拍图）。\n"
                "如果只是想先跑通流程，可加 --allow-synthetic-calib。")
        rng = np.random.default_rng(0)
        imgs = []
        for i in range(max(4, samples)):
            if i % 3 == 0:
                img = np.full((height, width, 3), 114, np.uint8)
            elif i % 3 == 1:
                img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            else:
                x = np.linspace(0, 255, width, dtype=np.uint8)
                img = np.stack([np.tile(x, (height, 1))] * 3, axis=-1)
            imgs.append(img)
        data = np.stack([np.transpose(i, (2, 0, 1))[None] for i in imgs])
        return data, "synthetic"

    imgs = []
    for i in range(samples):
        p = paths[i % len(paths)]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        imgs.append(img)
    if not imgs:
        raise SystemExit("校准集读取失败：没有可用图片")
    data = np.stack([np.transpose(i, (2, 0, 1))[None] for i in imgs])
    return data, "%d 张（%d 个文件，重复取样）" % (len(imgs), len(paths))


def simplify_onnx(model_file: str, input_shape, dump_dir: str) -> str:
    """用 onnxsim 固定输入 shape 并简化；失败则返回原文件。"""
    import onnx

    model = onnx.load(model_file)
    input_names = [i.name for i in model.graph.input]
    if not input_names:
        return model_file
    shapes = {input_names[0]: list(input_shape)}
    ok = False
    try:                       # 优先 onnxsim
        import onnxsim
        model, ok = onnxsim.simplify(model, input_shapes=shapes)
    except ImportError:
        try:                   # ultralytics 默认装的是 onnxslim
            import onnxslim
            model = onnxslim.slim(model)
            ok = True
        except ImportError:
            print("[warn] 未安装 onnxsim/onnxslim，跳过简化")
            return model_file
        except Exception as exc:
            print("[warn] onnxslim 简化失败：%s" % exc)
            return model_file
    except TypeError:
        try:
            model, ok = onnxsim.simplify(model, overwrite_input_shapes=shapes)
        except Exception as exc:
            print("[warn] onnxsim 简化失败：%s" % exc)
            return model_file
    except Exception as exc:
        print("[warn] onnxsim 简化失败：%s" % exc)
        return model_file
    if not ok:
        print("[warn] 简化校验失败，继续用原模型")
        return model_file
    out = os.path.join(dump_dir, "simplified.onnx")
    onnx.save(model, out)
    print("[*] ONNX 已简化 -> %s" % out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="YOLOv8 ONNX -> K230 kmodel")
    ap.add_argument("--onnx", required=True, help="输入 ONNX")
    ap.add_argument("--out", default=None, help="输出 kmodel，默认同名 .kmodel")
    ap.add_argument("--input-size", default="320", help="320 或 640x480")
    ap.add_argument("--calib-dir", default=None, help="校准图片目录")
    ap.add_argument("--samples", type=int, default=10, help="PTQ 校准张数")
    ap.add_argument("--target", default="k230")
    ap.add_argument("--quant", default="uint8", choices=["uint8", "int8", "int16"])
    ap.add_argument("--w-quant", default="uint8", choices=["uint8", "int8", "int16"])
    ap.add_argument("--calibrate", default="Kld", choices=["Kld", "NoClip"])
    ap.add_argument("--mean", default="0,0,0")
    ap.add_argument("--std", default="1,1,1")
    ap.add_argument("--input-range", default="0,1", help="uint8 反量化范围")
    ap.add_argument("--swap-rb", action="store_true", help="模型内部做 BGR->RGB")
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--allow-synthetic-calib", action="store_true")
    ap.add_argument("--dump-dir", default="tmp_nncase")
    ap.add_argument("--dump-ir", action="store_true")
    ap.add_argument("--ptq-profile", type=int, default=0, choices=[0, 1, 2, 3],
                    help="PTQ 档位：0=uint8/Kld，1=uint8/NoClip，2=Kld+MSE权重，3=Kld+MixQuant")
    ap.add_argument("--use-mse-quant-w", action="store_true")
    ap.add_argument("--use-mix-quant", action="store_true")
    args = ap.parse_args(argv)

    w, h = parse_size(args.input_size)
    w, h = align32(w), align32(h)
    input_shape = [1, 3, h, w]
    out = args.out or (str(Path(args.onnx).with_suffix(".kmodel")))
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    dotnet = _prepare_dotnet()
    if not dotnet:
        print("[warn] 没找到 DOTNET_ROOT；如果 import nncase 报 hostfxr 错误，请安装 .NET 7")
    try:
        import nncase
        import _nncase
        print("[*] nncase %s（.NET: %s）" % (
            getattr(_nncase, "__version__", "?"), dotnet or "?"))
    except Exception as exc:
        print("[x] 无法导入 nncase：%s" % exc)
        print("    安装：pip install nncase nncase-kpu；并配置 DOTNET_ROOT")
        return 2

    if not args.no_simplify:
        model_file = simplify_onnx(args.onnx, input_shape, str(dump_dir))
    else:
        model_file = args.onnx

    calib, desc = load_calib_images(args.calib_dir, w, h, args.samples,
                                    args.allow_synthetic_calib)
    print("[*] 校准集：%s" % desc)

    mean = [float(x) for x in args.mean.split(",")]
    std = [float(x) for x in args.std.split(",")]
    input_range = [float(x) for x in args.input_range.split(",")]

    opts = nncase.CompileOptions()
    opts.target = args.target
    opts.preprocess = True
    opts.swapRB = bool(args.swap_rb)
    opts.input_shape = input_shape
    opts.input_type = "uint8"
    opts.input_range = input_range
    opts.mean = mean
    opts.std = std
    opts.input_layout = "NCHW"
    opts.output_layout = "NCHW"
    opts.dump_ir = bool(args.dump_ir)
    opts.dump_asm = False
    opts.dump_dir = str(dump_dir)

    compiler = nncase.Compiler(opts)
    with open(model_file, "rb") as fp:
        compiler.import_onnx(fp.read(), nncase.ImportOptions())
    print("[*] ONNX 已导入，开始 PTQ…")

    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = int(calib.shape[0])
    ptq.quant_type = args.quant
    ptq.w_quant_type = args.w_quant
    ptq.calibrate_method = args.calibrate
    ptq.finetune_weights_method = "NoFineTuneWeights"
    if args.ptq_profile == 1:
        ptq.calibrate_method = "NoClip"
    elif args.ptq_profile == 2:
        try:
            ptq.use_mse_quant_w = True
        except Exception as exc:
            print("[warn] use_mse_quant_w 不可用：%s" % exc)
    elif args.ptq_profile == 3:
        try:
            ptq.use_mix_quant = True
        except Exception as exc:
            print("[warn] use_mix_quant 不可用：%s" % exc)
    if args.use_mse_quant_w:
        ptq.use_mse_quant_w = True
    if args.use_mix_quant:
        ptq.use_mix_quant = True
    print("[*] PTQ 档位 %d：quant=%s w_quant=%s calib=%s mse_w=%s mix=%s" % (
        args.ptq_profile, ptq.quant_type, ptq.w_quant_type,
        ptq.calibrate_method, getattr(ptq, "use_mse_quant_w", "?"),
        getattr(ptq, "use_mix_quant", "?")))
    # 官方示例的传参形式：List[List[np.ndarray]]，每项是 [1,3,H,W]
    ptq.set_tensor_data([[calib[i]] for i in range(calib.shape[0])])
    compiler.use_ptq(ptq)

    t0 = time.time()
    compiler.compile()
    kmodel = compiler.gencode_tobytes()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(kmodel)
    dt = time.time() - t0
    print("\n[✓] kmodel 已生成：%s" % out)
    print("    大小 %.2f MB，耗时 %.1fs，输入 %dx%d uint8" %
          (len(kmodel) / 1048576.0, dt, w, h))
    print("    下一步：python3 -m k230tool push %s --dest /sdcard/models/%s" %
          (out, Path(out).name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
