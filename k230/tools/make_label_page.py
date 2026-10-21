#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为某个采集目录生成可点击标注的网页：把帧列表和标注注入到 laser_label.html。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "tools" / "laser_label.html"


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python3 tools/make_label_page.py tools/laser_samples/xxx [out.html]")
        return 2
    d = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else d / "label.html"
    files = sorted((d / "raw").glob("frame_*.jpg"))
    items = [{"file": str(f.relative_to(d)), "url": "raw/" + f.name} for f in files]
    html = TPL.read_text(encoding="utf-8")
    html = html.replace("<script>",
                        "<script>window.FILES = %s;</script>\n<script>"
                        % json.dumps(items, ensure_ascii=False), 1)
    out.write_text(html, encoding="utf-8")
    print("标注页 ->", out)
    print("浏览器打开后：左键点光斑中心，无光斑按 N，导出后写回 %s" % (d / "annotation.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
