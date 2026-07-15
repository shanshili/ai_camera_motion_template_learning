"""
阶段 1：探测 (Probe)
====================
读输入元信息，建立统一时间轴，对照期望源规格做软校验（不一致只告警）。
"""

from ..stage import Stage, register
from ..ffio import ffprobe_meta
from ..timeline import Timeline


@register("probe")
class ProbeStage(Stage):
    name = "probe"

    def run(self, ctx):
        meta = ffprobe_meta(ctx.input_path)
        ctx.meta = meta
        ctx.timeline = Timeline(fps=meta["fps"], frame_count=meta["frame_count"])

        print(f"[probe] 输入: {meta['width']}x{meta['height']} "
              f"@ {float(meta['fps']):.3f}fps, {meta['frame_count']} 帧, "
              f"{meta['duration']:.3f}s, 音轨={'有' if meta['has_audio'] else '无'}")
        print(f"[probe] {ctx.timeline}")

        exp = ctx.config.get("source", {})
        if exp.get("width") and exp["width"] != meta["width"]:
            print(f"[probe][告警] 源宽 {meta['width']} != 期望 {exp['width']}")
        if exp.get("height") and exp["height"] != meta["height"]:
            print(f"[probe][告警] 源高 {meta['height']} != 期望 {exp['height']}")
        if exp.get("fps") and abs(float(meta["fps"]) - exp["fps"]) > 0.01:
            print(f"[probe][告警] 源帧率 {float(meta['fps']):.3f} != 期望 {exp['fps']}")

        out_w = ctx.config["output"]["width"]
        ratio = meta["width"] / out_w
        print(f"[probe] 分辨率预算: 源宽/成片宽 = {ratio:.2f}（≥2 才无损放大）")
        if ratio < 2.0:
            print("[probe][告警] 分辨率比 < 2，近景运镜会上采样、画质受损")
            print("             演示同分辨率时请设 camera.allow_upscale_for_demo=true")
        return ctx
