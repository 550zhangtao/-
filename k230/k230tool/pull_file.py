#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pull_file.py —— 从 K230 板子把文件拉回 PC
=========================================

板端没有现成的文件传输协议，这里用 raw REPL + base64 分块读：
    print('CHUNK', base64)  ->  PC 端解码拼接
实测约 20~25 KB/s，适合备份 kmodel / 标签 / 日志、抓板端截图。

用法：
    python3 -m k230tool pull /sdcard/models/x.kmodel ./x.kmodel
    python3 -m k230tool pull /sdcard/libs/YOLO.py ./YOLO.py
    python3 -m k230tool pull /sdcard/models -o ./models --list
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
import base64
import sys
import time
from pathlib import Path

from .link import K230Link, describe_ports, find_port

_PULL_SRC = """import binascii
_f = open({path!r}, 'rb')
_f.seek(0, 2)
_n = _f.tell()
_f.seek(0)
print('SIZE', _n)
while True:
    _b = _f.read({chunk})
    if not _b:
        break
    print('CHUNK', binascii.b2a_base64(_b).decode().strip())
_f.close()
print('DONE')
"""


def _upload(link: K230Link, src: str, step: int = 128, delay: float = 0.002):
    data = src.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
    for i in range(0, len(data), step):
        link.ser.write(data[i:i + step])
        time.sleep(delay)
    link.ser.write(b"\x04")


def pull(link: K230Link, remote: str, local: Path, chunk: int = 3072,
         timeout: float = 300.0, progress=None) -> int:
    """把板端 remote 拉到 local，返回字节数；失败抛 BoardError。"""
    remote = remote.rstrip("/")
    link.enter_raw()
    _upload(link, _PULL_SRC.format(path=remote, chunk=int(chunk)))

    buf = bytearray()
    got = bytearray()
    size = None
    t0 = time.monotonic()
    last_report = t0
    try:
        while time.monotonic() - t0 < timeout:
            data = link.ser.read(8192)
            if data:
                buf += data
            # 解析已经完整到达的行
            while b"\n" in buf:
                line, _, rest = buf.partition(b"\n")
                buf = bytearray(rest)
                text = line.decode("utf-8", "replace").strip().strip("\x04")
                if text.startswith("SIZE"):
                    try:
                        size = int(text.split()[1])
                    except (IndexError, ValueError):
                        pass
                elif text.startswith("CHUNK"):
                    try:
                        got += base64.b64decode(text.split(None, 1)[1])
                    except Exception:
                        pass
                    if progress and size:
                        now = time.monotonic()
                        if now - last_report >= 0.5:
                            last_report = now
                            progress(len(got), size)
                elif text == "DONE":
                    break
            else:
                continue
            break
    finally:
        link.exit_raw()

    if size is not None and len(got) != size:
        raise RuntimeError("大小不符：期望 %d，收到 %d" % (size, len(got)))
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(local.name + ".part")
    tmp.write_bytes(bytes(got))
    tmp.replace(local)
    return len(got)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="从 K230 拉文件")
    ap.add_argument("remote", help="板端路径")
    ap.add_argument("local", nargs="?", help="本地路径（默认用文件名）")
    ap.add_argument("--chunk", type=int, default=3072)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--port", default=None)
    args = ap.parse_args(argv)

    if not find_port():
        print("没有找到 K230：\n%s" % describe_ports())
        return 2
    local = Path(args.local or args.remote.rstrip("/").split("/")[-1])
    link = K230Link(port=args.port).open()
    try:
        def _progress(done, total):
            pct = 100.0 * done / total if total else 0
            sys.stderr.write("\r  %5.1f%%  %6.1f/%6.1f KB" %
                             (pct, done / 1024.0, total / 1024.0))
            sys.stderr.flush()
        n = pull(link, args.remote, local, chunk=args.chunk,
                 timeout=args.timeout, progress=_progress)
        sys.stderr.write("\n")
        print("[pull] %s -> %s（%d 字节）" % (args.remote, local, n))
        return 0
    except Exception as exc:
        print("\n[pull] 失败：%s" % exc, file=sys.stderr)
        return 1
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
