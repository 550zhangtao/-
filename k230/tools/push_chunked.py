#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分片推送大文件到板子（kmodel 这类 3MB+ 的二进制）。

单次 REPL 会话传 3MB 会中途丢字节（实测只写进去约 300KB），
所以切成 CHUNK 大小分片，每片单独一次写入，最后在板端按序合并。

用法：
    python3 tools/push_chunked.py big.kmodel --dest /sdcard/models/x.kmodel
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from k230tool.link import K230Link  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("local")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--chunk", type=int, default=196608, help="每片字节数")
    args = ap.parse_args()

    src = Path(args.local)
    data = src.read_bytes()
    total = len(data)
    n = (total + args.chunk - 1) // args.chunk
    print("[chunked] %s -> %s  (%d 字节, %d 片)" % (src.name, args.dest, total, n))

    link = K230Link()
    link.open()
    tmpdir = "/tmp_up"
    try:
        link.exec_and_collect(
            "import os\ntry:\n    os.mkdir('/%s')\nexcept Exception:\n    pass\nprint(1)"
            % tmpdir, timeout=15)
        # 先写到临时文件，校验大小通过后再 rename 覆盖目标。
        # 这样中途失败不会把原有可用文件写坏（踩过：直接把
        # /sdcard/models/target_pose.kmodel 覆盖成了 110KB 的残片）。
        tmp_dest = args.dest + ".part"
        for i in range(n):
            part = data[i * args.chunk:(i + 1) * args.chunk]
            mode = "wb" if i == 0 else "ab"
            # 单行：open + write + close 用分号拼接
            code = ("f=open('%s','%s');f.write(__import__('binascii')"
                    ".a2b_base64(b'%s'));f.close()" %
                    (tmp_dest, mode, base64.b64encode(part).decode()))
            out, err, ok = link.exec_and_collect(code, timeout=180)
            if not ok or b"Error" in out.encode() if isinstance(out, str) else False:
                print("  片 %d 写入失败: %s %s" % (i, out[:80], err[:120]))
                return 2
            if (i + 1) % 5 == 0 or i == n - 1:
                print("  片 %d/%d" % (i + 1, n), flush=True)
        out, err, ok = link.exec_and_collect(
            "import os;print(os.stat('%s')[6])" % tmp_dest, timeout=60)
        size = None
        for tok in out.replace("\r", " ").replace("\n", " ").split():
            if tok.isdigit():
                size = int(tok)
        print("[chunked] 板上大小 %s 字节（期望 %d）" % (size, total))
        if size != total:
            print("[chunked] !! 大小不符，保留原文件不动")
            return 3
        out, err, ok = link.exec_and_collect(
            "import os;os.rename('%s','%s');print(os.stat('%s')[6])"
            % (tmp_dest, args.dest, args.dest), timeout=60)
        print("[chunked] 已替换目标：%s" % out.strip())
        print("[chunked] 完成")
        return 0
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
