# -*- coding: utf-8 -*-
"""recover_deploy.py —— 一键恢复工作模式（小分块 + 校验 + 只读兜底）

2026-10-23 事故：`push_file` 默认 12 KB 分块推 18 KB 的 `gimbal_agent.py`
时，raw REPL 被大块写入拖挂（REPL 彻底无响应，Ctrl-C/Ctrl-D 无效），
同时 `/sdcard` 变成只读（OSError 30），只能硬件复位。

本脚本的做法：
  1. 等板子出现 /dev/ttyACM*，进 raw REPL；
  2. **2 KB 分块**写 `/sdcard/main.py.new`，板端 md5 校验；
  3. md5 一致才 rename 成 `/sdcard/main.py`（先删后改名，规避 EEXIST）；
  4. 若 `/sdcard` 只读（OSError 30），自动退到 `/data/main.py` 并在 REPL 里直接跑起来
     （会话内可用；持久化仍需修好 SD 卡）；
  5. 任何一步失败都**不再重试大块写入**，避免把板子再弄挂。

用法：
    python3 -m tools.recover_deploy                 # 用 board/gimbal_agent.py
    python3 -m tools.recover_deploy --run           # 部署后立即跑起来
    python3 -m tools.recover_deploy --src board/x.py --dest /sdcard/main.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k230tool.link import K230Link, BoardError, find_port  # noqa: E402

CHUNK = 2048          # 实测 12 KB 会把 raw REPL 拖挂；2 KB 稳定
STEP, DELAY = 128, 0.002


def board_write(link: K230Link, data: bytes, remote: str) -> bool:
    """用 base64 + 文本追加写文件（避开板端 'wb' 直写 OSError 30 的路径）。"""
    import base64
    b64_path = remote + ".b64"
    link.exec_and_collect("import os\ntry:\n    os.remove(%r)\nexcept Exception:\n    pass\n"
                          "print('CLEAN')" % b64_path, timeout=10)
    sent = 0
    for off in range(0, len(data), CHUNK):
        piece = data[off:off + CHUNK]
        code = ("f=open(%r,'a')\nf.write('%s')\nf.close()\nprint('W')\n"
                % (b64_path, base64.b64encode(piece).decode()))
        out, err, ok = link.exec_and_collect(code, timeout=25)
        if ("'W'" not in out) and ("W" not in out):
            print("  ! 写失败 @%d: %s %s" % (off, out[:120], err[:200]))
            return False
        sent += len(piece)
    # 板上解码成目标文件（这里才用 'wb'：小文件、一次成功即可）
    # 注意：板端 hashlib.md5().hexdigest() 不存在，要用 ubinascii.hexlify(digest())
    code = ("import ubinascii as B, os, hashlib\n"
            "r=open(%r,'r')\nd=r.read()\nr.close()\n"
            "w=open(%r,'wb')\nw.write(B.a2b_base64(d))\nw.close()\n"
            "h=hashlib.md5(open(%r,'rb').read())\n"
            "print('MD5', B.hexlify(h.digest()).decode())\n"
            "print('SZ', os.stat(%r)[6])\n"
            "try:\n    os.remove(%r)\nexcept Exception:\n    pass\n"
            % (b64_path, remote, remote, remote, b64_path))
    out, err, ok = link.exec_and_collect(code, timeout=60)
    want = hashlib.md5(data).hexdigest()
    ok_md5 = ("MD5 " + want) in out
    print("  校验: %s (%d 字节)" % ("md5 一致" if ok_md5 else "md5 不一致", len(data)))
    if not ok_md5:
        print("  out:", out[-300:], "err:", err[-300:])
    return ok_md5


def replace(link: K230Link, tmp: str, remote: str, size: int) -> bool:
    code = ("import os\n"
            "try:\n    os.remove(%r)\nexcept Exception:\n    pass\n"
            "try:\n    os.rename(%r, %r)\nexcept Exception as e:\n    print('MVERR', e)\n"
            "try:\n    print('SZ', os.stat(%r)[6])\n"
            "except Exception:\n    print('SZ', -1)\n" % (remote, tmp, remote, remote))
    out, err, ok = link.exec_and_collect(code, timeout=20)
    return ("SZ %d" % size) in out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "board" / "gimbal_agent.py"))
    ap.add_argument("--dest", default="/sdcard/main.py")
    ap.add_argument("--port", default=None)
    ap.add_argument("--run", action="store_true", help="部署后用 exec 立刻跑起来")
    ap.add_argument("--wait", type=float, default=60.0, help="等板子出现的最长秒数")
    args = ap.parse_args(argv)

    src = Path(args.src)
    data = src.read_bytes()
    print("[recover] %s -> %s（%d 字节）" % (src.name, args.dest, len(data)))

    # 等板子：不只看串口是否存在，还要能进 raw REPL（卡死时串口在但进不去）
    t0 = time.time()
    link = None
    port = None
    last_err = ""
    while time.time() - t0 < args.wait:
        cand = args.port or find_port()
        if not cand:
            time.sleep(1.0)
            continue
        try:
            link = K230Link(port=cand)
            link.open()
            link.enter_raw(timeout=3.0)
            port = cand
            break
        except Exception as e:
            last_err = str(e)[:80]
            try:
                if link is not None:
                    link.close()
            except Exception:
                pass
            link = None
            time.sleep(2.0)
    if link is None:
        print("[recover] %.0fs 内板子没恢复（%s）。请按 K230 的 RESET 键，"
              "或拔插 K230 的 USB 线后重跑本脚本。" % (args.wait, last_err))
        return 2
    print("[recover] 串口:", port, "（raw REPL 就绪）")
    try:
        # 1) 先探一下 /sdcard 是否可写
        out, err, ok = link.exec_and_collect(
            "import os\n"
            "try:\n    f=open('/sdcard/_rwtest','w')\n    f.write('x')\n    f.close()\n"
            "    os.remove('/sdcard/_rwtest')\n    print('SD_RW')\n"
            "except Exception as e:\n    print('SD_RO', repr(e))\n", timeout=15)
        sd_rw = "SD_RW" in out
        print("[recover] /sdcard:", "可写" if sd_rw else "只读（%s）" % out.strip()[:80])

        dest = args.dest
        if not sd_rw and dest.startswith("/sdcard/"):
            dest = "/data/agent.py"   # 实测 /data/main.py 会触发固件 watcher 把板子搞卡死
            print("[recover] 退到 %s（会话内可用；持久化需修 SD）" % dest)

        tmp = dest + ".new"
        if not board_write(link, data, tmp):
            print("[recover] 写入失败，停止（不再重试大块写，避免拖挂 REPL）")
            return 1
        if not replace(link, tmp, dest, len(data)):
            print("[recover] 替换失败：%s 仍在，可手工 mv" % tmp)
            return 1
        print("[recover] 部署成功:", dest)

        if args.run:
            # 必须在普通 REPL 里 exec：raw REPL 会被脚本占死，退不出来
            link.exit_raw()
            time.sleep(0.3)
            link.ser.write(("exec(open(%r).read())\r\n" % dest).encode())
            print("[recover] 已启动 %s（REPL 会被代理占用，属正常）" % dest)

        # 工作模式：删 offline_off，让 /sdcard/main.py 上电自跑
        if dest.startswith("/sdcard/"):
            link.exec_and_collect(
                "import os\ntry:\n    os.remove('/sdcard/offline_off')\n    print('CLEARED')\n"
                "except Exception:\n    print('NO_MARKER')\n", timeout=10)
            print("[recover] 已清 offline_off；复位后即为工作模式")
    except BoardError as e:
        print("[recover] 板端错误:", e)
        return 1
    finally:
        try:
            link.exit_raw()
        except Exception:
            pass
        link.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
