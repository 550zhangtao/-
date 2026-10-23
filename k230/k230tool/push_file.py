#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
push_file.py —— 通过 USB 往 K230 的 /sdcard 传文件
========================================================

板子上要长期用的脚本、库、配置都可以这样推过去，不必打开 CanMV IDE 手动拖拽。

原理：raw REPL 里分块执行 `open(path,'ab').write(b'...')`。
板端 UART RX 会因突发丢字节（实测），所以按 1KB 分块 + 小延时。

用法：
    python3 -m k230tool push my_script.py --dest /sdcard/my_script.py
    python3 -m k230tool push --dest /sdcard/libs 某个目录
    python3 -m k230tool push --list /sdcard
    python3 -m k230tool push --mkdir /sdcard/models
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
import os
import sys
import time
from pathlib import Path

from .link import K230Link, describe_ports, find_port  # noqa: E402

CHUNK = 12288            # 每次 exec 写入的原始字节数（base64 后约 16KB 文本）
_UPLOAD_STEP = 128       # 板端 RX 缓冲有限：实测 128B/2ms 不丢字节
_UPLOAD_DELAY = 0.002


def _write_code(link: K230Link, code: str) -> None:
    """把一段代码按节流方式灌进 raw REPL，并以 \x04 结束。"""
    data = code.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
    for i in range(0, len(data), _UPLOAD_STEP):
        link.ser.write(data[i:i + _UPLOAD_STEP])
        time.sleep(_UPLOAD_DELAY)
    link.ser.write(b"\x04")


def _read_repl(link: K230Link, timeout: float = 15.0) -> str:
    buf = bytearray()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        chunk = link.ser.read(4096)
        if chunk:
            buf += chunk
            if bytes(buf[-4:]).find(b"\x04>") >= 0:
                break
        elif bytes(buf[-3:]).endswith(b">"):
            break
    return bytes(buf).decode("utf-8", "replace")


def push_one(link: K230Link, local: str, remote: str, verbose: bool = True) -> bool:
    """单次 raw REPL 会话内用 base64 分块写文件（比逐字节 \\xNN 快 3 倍）。"""
    import base64

    size = os.path.getsize(local)
    if verbose:
        print("[*] %s -> %s (%.1f KB)" % (local, remote, size / 1024.0))
    link.enter_raw()
    parent = remote.rsplit("/", 1)[0] if "/" in remote else ""
    if parent:
        _write_code(link, "import os\ntry:\n    os.mkdir(%r)\nexcept Exception:\n    pass\n"
                          "import binascii\nf=open(%r,'wb')\nf.close()\nprint('OK')"
                          % (parent, remote))
    else:
        _write_code(link, "import binascii\nf=open(%r,'wb')\nf.close()\nprint('OK')" % remote)
    if "OK" not in _read_repl(link, 15):
        print("  ! 无法创建目标文件")
        return False

    sent = 0
    t0 = time.monotonic()
    with open(local, "rb") as fp:
        while True:
            chunk = fp.read(CHUNK)
            if not chunk:
                break
            b64 = base64.b64encode(chunk).decode("ascii")
            code = ("f=open(%r,'ab')\nf.write(binascii.a2b_base64(b'%s'))\n"
                    "f.close()\nprint('W')\n" % (remote, b64))
            _write_code(link, code)
            resp = _read_repl(link, 20)
            if "'W'" not in resp and "W" not in resp:
                print("  ! 写入失败 @%d: %s" % (sent, resp[:200]))
                return False
            sent += len(chunk)
            if verbose and sent % (CHUNK * 8) == 0:
                el = time.monotonic() - t0
                print("  %5.1f%%  %6.1f KB  %5.1f KB/s"
                      % (100.0 * sent / size, sent / 1024.0,
                         sent / 1024.0 / max(el, 1e-3)))

    _write_code(link, "import os\nprint('SZ', os.stat(%r)[6])" % remote)
    resp = _read_repl(link, 15)
    ok_size = ("SZ %d" % size) in resp
    link.exit_raw()
    print("  完成: %d/%d 字节 %s" % (sent, size, "✓" if ok_size else "✗ 大小不符"))
    return ok_size


def main() -> int:
    ap = argparse.ArgumentParser(description="往 K230 SD 卡推文件")
    ap.add_argument("src", nargs="?", help="本地文件或目录")
    ap.add_argument("--dest", default=None, help="目标路径（目录或文件名）")
    ap.add_argument("--list", default=None, help="列出板端目录")
    ap.add_argument("--mkdir", default=None, help="在板端建目录")
    args = ap.parse_args()

    if not find_port():
        print("没有找到 K230：\n%s" % describe_ports())
        return 2
    link = K230Link(port=find_port())
    link.open()
    try:
        if args.list:
            out, err, ok = link.exec_and_collect(
                "import os\nprint('LS', os.listdir(%r))\n" % args.list,
                timeout=15)
            print(out.strip() or err)
            return 0
        if args.mkdir:
            out, err, ok = link.exec_and_collect(
                "import os\ntry:\n    os.mkdir(%r)\n    print('OK')\n"
                "except Exception as e:\n    print('EXIST')\n" % args.mkdir,
                timeout=10)
            print(args.mkdir, out.strip() or err)
            return 0
        if not args.src:
            ap.print_help()
            return 1

        src = args.src.rstrip("/")
        if os.path.isdir(src):
            dest_dir = args.dest or ("/sdcard/" + os.path.basename(src))
            link.exec_and_collect(
                "import os\ntry:\n    os.mkdir(%r)\nexcept Exception:\n    pass\n"
                "print('OK')\n" % dest_dir, timeout=10)
            n_ok = 0
            for name in sorted(os.listdir(src)):
                p = os.path.join(src, name)
                if not os.path.isfile(p):
                    continue
                if push_one(link, p, dest_dir + "/" + name):
                    n_ok += 1
            print("[*] 目录完成：%d 个文件" % n_ok)
            return 0 if n_ok else 1

        dest = args.dest or ("/sdcard/" + os.path.basename(src))
        return 0 if push_one(link, src, dest) else 1
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
