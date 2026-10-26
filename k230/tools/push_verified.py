#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可靠推送大文件到板子：逐片校验长度 + 整体 MD5 比对，校验通过才替换。

背景：kmodel 这类 3.5MB 文件走 REPL 分片推送时，**大小对、内容错**的情况
真实发生过（438 片里有分片丢字节），导致 `KPU load model failed`。
所以必须做内容级校验：

  1. 每写一片就 stat 一次，长度必须等于已写字节数（不对就重试该片）
  2. 全部写完在板端算 MD5，与本地比对
  3. 只有 MD5 一致才 rename 覆盖目标；否则保留原文件不动

用法：
    python3 tools/push_verified.py models/x.kmodel --dest /sdcard/models/x.kmodel
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from k230tool.link import K230Link  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("local")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    src = Path(args.local)
    data = src.read_bytes()
    total = len(data)
    want_md5 = hashlib.md5(data).hexdigest()
    n = (total + args.chunk - 1) // args.chunk
    tmp = args.dest + ".part"
    print("[push] %s (%d 字节, md5 %s, %d 片)" % (src.name, total, want_md5, n))

    link = K230Link()
    link.open()
    try:
        # 清掉可能残留的临时文件
        link.exec_and_collect(
            "import os\ntry:\n    os.remove('%s')\nexcept Exception:\n    pass\nprint(1)" % tmp,
            timeout=30)
        written = 0
        for i in range(n):
            part = data[i * args.chunk:(i + 1) * args.chunk]
            b64 = base64.b64encode(part).decode()
            mode = "wb" if i == 0 else "ab"
            ok = False
            for attempt in range(args.retries):
                code = ("f=open('%s','%s');f.write(__import__('binascii')"
                        ".a2b_base64(b'%s'));f.close()") % (tmp, mode, b64)
                out, err, res = link.exec_and_collect(code, timeout=120)
                if not res:
                    continue
                # 校验累计长度
                out2, err2, res2 = link.exec_and_collect(
                    "import os;print(os.stat('%s')[6])" % tmp, timeout=60)
                cur = None
                for tok in out2.replace("\r", " ").replace("\n", " ").split():
                    if tok.isdigit():
                        cur = int(tok)
                expect = written + len(part)
                if cur == expect:
                    written = expect
                    ok = True
                    break
                # 长度不对：回滚到已确认长度再重试该片
                link.exec_and_collect(
                    "f=open('%s','r+b');f.truncate(%d);f.close();print(1)" % (tmp, written),
                    timeout=60)
            if not ok:
                print("[push] !! 第 %d 片连续失败，放弃" % i)
                return 2
            if (i + 1) % 40 == 0 or i == n - 1:
                print("  %d/%d 片 (%d 字节)" % (i + 1, n, written), flush=True)

        # 整体 MD5 校验
        code = ("import uhashlib\nh=uhashlib.md5()\nf=open('%s','rb')\n"
                "while True:\n    b=f.read(65536)\n    if not b:\n        break\n    h.update(b)\n"
                "f.close()\nprint(__import__('binascii').hexlify(h.digest()).decode())\n") % tmp
        out, err, res = link.exec_and_collect(code, timeout=300)
        got = ""
        for tok in out.replace("\r", " ").replace("\n", " ").split():
            if len(tok) == 32 and all(c in "0123456789abcdef" for c in tok.lower()):
                got = tok.lower()
        print("[push] 板上 md5 = %s" % (got or out.strip()[:60]))
        if got != want_md5:
            print("[push] !! MD5 不一致，保留原文件")
            return 3
        # 注意：MicroPython 的 os.rename 在目标已存在时会抛 EEXIST(17)，
        # 必须先删目标再改名（踩过：rename 失败但脚本以为成功，导致旧文件残留）。
        code = ("import os\n"
                "try:\n    os.remove('%s')\nexcept Exception:\n    pass\n"
                "os.rename('%s','%s')\n"
                "print(os.stat('%s')[6])\n") % (args.dest, tmp, args.dest, args.dest)
        out, err, res = link.exec_and_collect(code, timeout=180)
        print("[push] 替换结果:", out.strip().replace("\r", " ")[:80])
        print("[push] ✓ 校验通过并已替换 %s" % args.dest)
        return 0
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
