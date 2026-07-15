#!/usr/bin/env python3
"""
AI 运镜生成入口
===============
读预设 YAML，对 io.input 下的视频跑「感知→主体→分镜→相机→渲染」，输出运镜视频。

用法：
  python run.py                                  # 处理 io.input 目录所有视频
  python run.py --input dataset/demo.mp4 --output output/demo_ai.mp4
  python run.py --config config/default.yaml

注意：多人骨架由独立前端 multi_person_tracking 离线产出 tracked_keypoints.json，
      放到对应 analysis 子目录后，本入口以 reuse_existing 复用它。见 README。
"""

import argparse
import os
import sys
import traceback

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
from pipeline_3 import run_pipeline  # noqa: E402

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def _abs(path, base="."):
    path = str(path).replace("\\", os.sep)
    if os.path.isabs(path):
        return os.path.abspath(path)
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base, path))


def _list_videos(path):
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith(VIDEO_EXTS))
    return []


def _run_one(config, in_path, out_path):
    cfg = dict(config)
    cfg["io"] = dict(cfg.get("io", {}))
    cfg["io"]["input"] = in_path
    cfg["io"]["output"] = out_path
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"\n########## {os.path.basename(in_path)} -> {os.path.basename(out_path)} ##########")
    run_pipeline(cfg)


def main():
    ap = argparse.ArgumentParser(description="AI 运镜生成")
    ap.add_argument("--input", default=None, help="输入视频文件或目录（缺省用 io.input）")
    ap.add_argument("--output", default=None, help="输出视频文件或目录（缺省用 io.output）")
    ap.add_argument("--config", default="config/default.yaml", help="配置文件")
    ap.add_argument("--template", default=None,
                    help="模板 json 路径（覆盖 config 的 template.path）；不给则用 config 里的，"
                         "config 也为空则用内置默认模板")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.template:
        config["template"] = dict(config.get("template", {}))
        config["template"]["path"] = _abs(args.template)
    tpath = (config.get("template") or {}).get("path")
    print(f"[run] 模板: {tpath if tpath else '(内置默认模板 generic_v1)'}", flush=True)

    in_spec = _abs(args.input or config.get("io", {}).get("input", "./dataset/"))
    out_spec = _abs(args.output or config.get("io", {}).get("output", "./output/"))

    print(f"[run] 配置: {os.path.abspath(args.config)}", flush=True)
    print(f"[run] 输入: {in_spec}", flush=True)
    print(f"[run] 输出: {out_spec}", flush=True)

    videos = _list_videos(in_spec)
    print(f"[run] 扫描到 {len(videos)} 个视频"
          + ("：" + ", ".join(os.path.basename(v) for v in videos) if videos else ""),
          flush=True)
    if not videos:
        print(f"[run][错误] 在 {in_spec} 未找到视频"
              f"（支持 {', '.join(VIDEO_EXTS)}）。\n"
              f"        请检查 config 的 io.input，或用 --input 指定文件/目录。", flush=True)
        sys.exit(2)

    if len(videos) == 1 and os.path.splitext(out_spec)[1]:
        _run_one(config, videos[0], out_spec)
        return

    os.makedirs(out_spec, exist_ok=True)
    for k, v in enumerate(videos, 1):
        print(f"\n[run] ===== 视频 {k}/{len(videos)} =====", flush=True)
        stem = os.path.splitext(os.path.basename(v))[0]
        out_path = os.path.join(out_spec, f"{stem}_ai.mp4")
        cfg = dict(config)
        cfg["analysis"] = dict(config.get("analysis", {}))
        base_out = cfg["analysis"].get("out_dir", "./analysis_out")
        cfg["analysis"]["out_dir"] = base_out
        _run_one(cfg, v, out_path)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("\n[run][异常] 管线中断，堆栈如下：", flush=True)
        traceback.print_exc()
        sys.exit(1)
