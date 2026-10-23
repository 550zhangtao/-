# -*- coding: utf-8 -*-
"""lean_recover.py —— 最小恢复：让板上代理重新跑起来（不写 /sdcard）

适用：`/sdcard` 只读、`mode.sh work` 写不进 `main.py`、REPL 又脆弱的时候。

做法（两步，都是小写入）：
  1. 在 /data 写一个 ~200 字节的引导文件（不碰 `/sdcard`，**绝不写 `/data/main.py`**——
     实测那个名字会触发固件 watcher 把整板卡死）；
  2. 在**普通 REPL**里 `exec(open('/data/boot_agent_v2.py').read())`。

为什么引导文件要重定向 stdout（2026-10-23 复盘）：
  代理每 60 帧 `print` 一次；PC 侧没人读 USB CDC 时写缓冲迟早填满，
  板端 print 永久阻塞 -> 整个 VM 卡死（现象是连打开 /dev/ttyACM0 都卡住）。
  把 stdout 写到 `/data/agent.log` 后，USB 上不再有周期性输出，板子就稳了。

用法：
    python3 -m tools.lean_recover --wait 60     # 等板子 -> 引导 -> 启动
    python3 -m tools.lean_recover --status
    python3 -m tools.lean_recover --log         # 看 /data/agent.log 尾部
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k230tool.link import K230Link, find_port  # noqa: E402

BOOT_PATH = "/data/boot_agent_v2.py"
BOOT_SRC = (
    "import sys\n"
    "try:\n"
    "    sys.stdout = open('/data/agent.log', 'w')\n"
    "except Exception:\n"
    "    pass\n"
    "exec(open('/sdcard/gimbal_agent.py').read())\n"
)


def wait_repl(seconds: float):
    t0 = time.time()
    last = ""
    while time.time() - t0 < seconds:
        port = find_port()
        if not port:
            time.sleep(1.0)
            continue
        link = None
        try:
            link = K230Link(port=port)
            link.open()
            link.enter_raw(timeout=3.0)
            print("[lean] REPL 就绪:", port)
            return link
        except Exception as e:
            last = str(e)[:80]
            try:
                if link is not None:
                    link.close()
            except Exception:
                pass
            time.sleep(2.0)
    print("[lean] %.0fs 内没等到 REPL（%s）" % (seconds, last))
    return None


def ensure_boot(link) -> bool:
    """写引导文件；已存在且内容正确就复用（覆盖已存在文件可能 EPERM）。"""
    code = ("import os\n"
            "p=%r\n"
            "ok=False\n"
            "try:\n"
            "    body=open(p).read()\n"
            "    ok=('gimbal_agent' in body) and ('agent.log' in body)\n"
            "except Exception:\n"
            "    ok=False\n"
            "print('HAS', ok)\n"
            "if not ok:\n"
            "    try:\n"
            "        os.remove(p)\n"
            "    except Exception:\n"
            "        pass\n"
            "    try:\n"
            "        open(p,'w').write(%r)\n"
            "        print('WROTE', len(open(p).read()))\n"
            "    except Exception as e:\n"
            "        print('WERR', repr(e))\n" % (BOOT_PATH, BOOT_SRC))
    out, err, ok = link.exec_and_collect(code, timeout=20)
    good = ("HAS True" in out) or ("WROTE" in out)
    if not good:
        print("[lean] 引导文件失败:", (out or "")[:160], (err or "")[:200])
    return good


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=float, default=120.0)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args(argv)

    if args.status or args.log:
        port = find_port()
        if not port:
            print("[lean] 没有 K230 串口")
            return 2
        link = K230Link(port=port)
        link.open()
        try:
            code = "print('PORT_OK')\n"
            if args.log:
                code += ("try:\n"
                         "    t=open('/data/agent.log').read()\n"
                         "    print('LOGTAIL', t[-600:])\n"
                         "except Exception as e:\n"
                         "    print('NOLOG', e)\n")
            out, err, ok = link.exec_and_collect(code, timeout=10)
            print(out or err)
        finally:
            link.exit_raw()
            link.close()
        return 0

    link = wait_repl(args.wait)
    if link is None:
        return 2
    try:
        if not ensure_boot(link):
            return 1
        link.exit_raw()                      # 必须在普通 REPL 里 exec
        time.sleep(0.3)
        link.ser.write(("exec(open(%r).read())\r\n" % BOOT_PATH).encode())
        print("[lean] 已启动代理（stdout -> /data/agent.log）")
        time.sleep(2.0)
        try:
            buf = link.ser.read(4096)
        except Exception:
            buf = b""
        if buf:
            print("   ", buf.decode("utf-8", "replace").replace("\r", " ")[:200])
        return 0
    finally:
        try:
            link.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
