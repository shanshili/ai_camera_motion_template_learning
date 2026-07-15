"""
阶段 5：渲染 (Render) · 三路输出 + 底部动态音乐能量时间轴
====================================================

输出：
  - final         ：最终运镜视频，可在底部追加动态音乐能量时间轴
  - skeleton_raw ：骨架标记原始视频
  - skeleton_cam ：骨架标记运镜视频

本版本特点：
  - 时间轴配置直接写在本文件顶部，不需要单独在 YAML 里配置 render.timeline。
  - 默认主画面按 1280x720 输出。
  - 底部追加音乐能量时间轴，不遮挡主画面。
  - 如果主画面 1280x720，时间轴 128px，则 final 实际为 1280x848。
  - fps 仍然使用 ctx.timeline.fps，当前你的管线是 30fps。
"""

import json
import math
import os

import cv2
import numpy as np

from ..stage import Stage, register
from ..ffio import frame_reader, FrameWriter
from ..transform import camera_matrix, apply_to_frame, effective_max_zoom
from .. import annotate as ann
from .. import skeleton as sk
from ..log import log, Progress


# ============================================================
# 这里就是写死在代码里的配置
# ============================================================

DEFAULT_OUTPUT_WIDTH = 1280
DEFAULT_OUTPUT_HEIGHT = 720

# 默认只输出 final。如果你还想同时输出骨架视频，可以改成：
# DEFAULT_RENDER_OUTPUTS = ["final", "skeleton_raw", "skeleton_cam"]
# debug = 带 C位骨架/预C位标注 + 景别字幕 + 分镜时间轴的排查版
DEFAULT_RENDER_OUTPUTS = ["final", "debug"]

DEFAULT_MOTION_BLUR_CONFIG = {
    "enabled": True,
    "threshold_px": 14.0,
}

DEFAULT_SKELETON_CONFIG = {
    "line_width": 4,
    "point_radius": 5,
    "min_confidence": 0.2,
    "background": "video",
}

# 动态音乐能量时间轴配置
TIMELINE_CONFIG = {
    "enabled": True,

    # append：不遮挡主画面，直接把时间轴拼到画面下方
    # 当前代码实现的是 append 模式
    "mode": "append",

    # 1280x720 横屏建议 96～140
    "height": 128,

    # 时间轴左右留白
    "pad_x": 52,

    # 时间轴内部上下留白
    "pad_top": 18,
    "pad_bottom": 30,

    # 曲线宽度
    "line_width": 2,

    # 文字大小
    "font_scale": 0.55,

    # 是否显示拍点、强拍、段落背景
    "show_beats": True,
    "show_sections": True,
    "section_alpha": 0.55,

    # 颜色，格式是 RGB 十六进制
    "bg": "#0b0d12",
    "grid_color": "#303642",
    "curve_color": "#596272",
    "played_color": "#22d3ee",
    "beat_color": "#ffffff",
    "downbeat_color": "#ff4d6d",
    "text_color": "#e5e7eb",
}


# ============================================================
# 原有运动模糊 / 输出路径
# ============================================================

def _directional_blur(img, dx, dy, strength=1.0):
    mag = float(math.hypot(dx, dy)) * float(strength)
    if mag < 2.0:
        return img

    k = int(max(3, min(15, round(mag / 3.0))))
    if k % 2 == 0:
        k += 1

    kernel = np.zeros((k, k), dtype=np.float32)
    c = k // 2

    if abs(dx) >= abs(dy):
        kernel[c, :] = 1.0
    else:
        kernel[:, c] = 1.0

    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def _out_paths(base_out):
    root, ext = os.path.splitext(base_out)
    ext = ext or ".mp4"
    return {
        "final": base_out,
        "debug": f"{root}_debug{ext}",          # C位/预C位标注 + 景别字幕 + 分镜时间轴
        "skeleton_raw": f"{root}_skeleton_raw{ext}",
        "skeleton_cam": f"{root}_skeleton_cam{ext}",
    }


# ============================================================
# 音乐能量时间轴相关函数
# ============================================================

def _try_load_json(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _get_music_result(ctx):
    """
    查找音乐分析结果。

    期望结构类似：
    {
        "energy_curve": [0.1, 0.3, 0.7, ...],
        "beat_grid": [
            {"frame": 12, "t": 0.4, "is_downbeat": false},
            ...
        ],
        "sections": [
            {"start_f": 0, "end_f": 120, "label": "mid"},
            ...
        ]
    }

    注意：
    这里读取的是音乐分析阶段产物，不是读取配置文件。
    如果你的音乐分析 JSON 文件名不同，可以把文件名加到 candidates 里。
    """

    # 1. 优先从 ctx.extras 里找
    for key in (
        "music",
        "music_result",
        "music_analysis",
        "analysis_music",
    ):
        value = ctx.extras.get(key)
        if isinstance(value, dict):
            return value

    # 2. 再从 ctx.extras 里记录的路径找
    for key in (
        "music_path",
        "music_json_path",
        "music_analysis_path",
    ):
        value = _try_load_json(ctx.extras.get(key))
        if isinstance(value, dict):
            return value

    # 3. 最后从 analysis.out_dir 里按常见文件名找
    out_dir = ctx.config.get("analysis", {}).get("out_dir")
    if out_dir:
        candidates = (
            "music.json",
            "music_result.json",
            "music_analysis.json",
            "analysis_music.json",
        )
        for name in candidates:
            value = _try_load_json(os.path.join(out_dir, name))
            if isinstance(value, dict):
                return value

    return None


def _resample_energy_curve(music, frame_count):
    """
    把音乐能量曲线重采样到视频总帧数。

    这样可以保证：
      energy_curve[frame_idx] 和当前渲染帧严格对齐。
    """
    if not music:
        return np.zeros(frame_count, dtype=np.float32)

    energy = music.get("energy_curve", [])
    energy = np.asarray(energy, dtype=np.float32)

    if energy.size == 0:
        return np.zeros(frame_count, dtype=np.float32)

    energy = np.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

    if energy.size != frame_count:
        if energy.size == 1:
            energy = np.repeat(energy, frame_count)
        else:
            old_x = np.linspace(0, frame_count - 1, energy.size)
            new_x = np.arange(frame_count)
            energy = np.interp(new_x, old_x, energy).astype(np.float32)

    # 百分位归一化，避免单个极端峰值把曲线压扁
    lo, hi = np.percentile(energy, [2, 98])
    if hi > lo:
        energy = (energy - lo) / (hi - lo)

    return np.clip(energy, 0.0, 1.0).astype(np.float32)


def _hex_to_bgr(hex_color, default):
    """
    配置颜色用 #RRGGBB。
    OpenCV 绘制需要 BGR。
    """
    if not isinstance(hex_color, str):
        return default

    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        return default

    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return b, g, r
    except ValueError:
        return default


def _fmt_time(seconds):
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def _prepare_music_timeline(ctx, width, video_height):
    """
    预计算底部音乐能量时间轴的数据。

    width       ：最终主画面宽度，比如 1280
    video_height：最终主画面高度，比如 720

    输出的 final 高度会是：
      video_height + timeline.height
    """
    timeline_cfg = TIMELINE_CONFIG

    if not timeline_cfg.get("enabled", False):
        return None

    frame_count = int(ctx.timeline.frame_count)
    if frame_count <= 0:
        return None

    music = _get_music_result(ctx) or {}
    if not music:
        print("[render][timeline][告警] 未找到音乐分析结果，底部时间轴会显示为空能量曲线")

    panel_h = int(timeline_cfg.get("height", 128))
    panel_h = max(48, panel_h)

    # libx264 + yuv420p 通常要求高度为偶数
    if (video_height + panel_h) % 2 != 0:
        panel_h += 1

    pad_x = int(timeline_cfg.get("pad_x", max(40, round(width * 0.04))))
    pad_top = int(timeline_cfg.get("pad_top", 18))
    pad_bottom = int(timeline_cfg.get("pad_bottom", 30))

    graph_h = max(24, panel_h - pad_top - pad_bottom)

    x0 = pad_x
    x1 = width - pad_x
    y_top = pad_top
    y_base = pad_top + graph_h

    energy = _resample_energy_curve(music, frame_count)

    xs = np.linspace(x0, x1, frame_count).round().astype(np.int32)
    ys = (y_base - energy * graph_h).round().astype(np.int32)
    energy_pts = np.stack([xs, ys], axis=1).astype(np.int32)

    fps_float = float(ctx.timeline.fps)

    beats = []
    for beat in music.get("beat_grid", []):
        f = beat.get("frame")

        if f is None and beat.get("t") is not None:
            f = int(round(float(beat["t"]) * fps_float))

        if f is None:
            continue

        f = max(0, min(frame_count - 1, int(f)))
        x = int(round(x0 + (x1 - x0) * f / max(1, frame_count - 1)))

        beats.append({
            "frame": f,
            "x": x,
            "is_downbeat": bool(beat.get("is_downbeat", False)),
        })

    sections = []
    for section in music.get("sections", []):
        start_f = int(section.get("start_f", 0))
        end_f = int(section.get("end_f", frame_count - 1))

        start_f = max(0, min(frame_count - 1, start_f))
        end_f = max(start_f + 1, min(frame_count, end_f))

        sx0 = int(round(x0 + (x1 - x0) * start_f / max(1, frame_count - 1)))
        sx1 = int(round(x0 + (x1 - x0) * end_f / max(1, frame_count - 1)))

        sections.append({
            "x0": sx0,
            "x1": sx1,
            "label": section.get("label", "mid"),
        })

    return {
        "height": panel_h,
        "width": width,
        "frame_count": frame_count,
        "x0": x0,
        "x1": x1,
        "y_top": y_top,
        "y_base": y_base,
        "graph_h": graph_h,
        "energy_pts": energy_pts,
        "beats": beats,
        "sections": sections,
        "cfg": timeline_cfg,
    }


def _draw_music_timeline(frame, timeline, frame_idx, fps):
    """
    把动态音乐能量时间轴追加到画面下方。

    重点：
      使用 np.vstack([frame, panel])
      所以它不会遮挡原运镜画面。
    """
    if timeline is None:
        return frame

    h, w = frame.shape[:2]
    panel_h = int(timeline["height"])
    cfg = timeline["cfg"]

    bg_color = _hex_to_bgr(cfg.get("bg", "#0b0d12"), (12, 14, 18))
    grid_color = _hex_to_bgr(cfg.get("grid_color", "#303642"), (66, 54, 48))
    curve_color = _hex_to_bgr(cfg.get("curve_color", "#596272"), (114, 98, 89))
    played_color = _hex_to_bgr(cfg.get("played_color", "#22d3ee"), (238, 211, 34))
    beat_color = _hex_to_bgr(cfg.get("beat_color", "#ffffff"), (230, 230, 230))
    downbeat_color = _hex_to_bgr(cfg.get("downbeat_color", "#ff4d6d"), (109, 77, 255))
    text_color = _hex_to_bgr(cfg.get("text_color", "#e5e7eb"), (235, 231, 229))

    panel = np.full((panel_h, w, 3), bg_color, dtype=np.uint8)

    x0 = int(timeline["x0"])
    x1 = int(timeline["x1"])
    y_top = int(timeline["y_top"])
    y_base = int(timeline["y_base"])
    graph_h = int(timeline["graph_h"])

    # 段落背景
    if cfg.get("show_sections", True):
        section_colors = {
            "low": (28, 44, 36),
            "mid": (42, 40, 28),
            "high": (48, 30, 36),
        }

        overlay = panel.copy()

        for section in timeline["sections"]:
            color = section_colors.get(section["label"], (36, 36, 36))
            cv2.rectangle(
                overlay,
                (int(section["x0"]), 0),
                (int(section["x1"]), panel_h),
                color,
                -1,
            )

        alpha = float(cfg.get("section_alpha", 0.55))
        panel = cv2.addWeighted(overlay, alpha, panel, 1.0 - alpha, 0)

    # 网格 / 参考线
    cv2.line(panel, (x0, y_top), (x1, y_top), grid_color, 1, cv2.LINE_AA)
    cv2.line(panel, (x0, y_base), (x1, y_base), grid_color, 1, cv2.LINE_AA)
    cv2.line(
        panel,
        (x0, y_top + graph_h // 2),
        (x1, y_top + graph_h // 2),
        grid_color,
        1,
        cv2.LINE_AA,
    )

    # 拍点 / 强拍
    if cfg.get("show_beats", True):
        for beat in timeline["beats"]:
            bx = int(beat["x"])
            is_downbeat = bool(beat["is_downbeat"])

            by0 = y_top if is_downbeat else int(y_base - graph_h * 0.42)
            by1 = min(panel_h - 24, y_base + 6)

            cv2.line(
                panel,
                (bx, by0),
                (bx, by1),
                downbeat_color if is_downbeat else beat_color,
                2 if is_downbeat else 1,
                cv2.LINE_AA,
            )

    # 整首音乐能量曲线，灰色底图
    pts = timeline["energy_pts"]
    if len(pts) > 1:
        cv2.polylines(panel, [pts], False, curve_color, 1, cv2.LINE_AA)

    # 已播放部分，高亮
    played_n = max(1, min(int(frame_idx) + 1, len(pts)))
    if played_n > 1:
        cv2.polylines(
            panel,
            [pts[:played_n]],
            False,
            played_color,
            int(cfg.get("line_width", 2)),
            cv2.LINE_AA,
        )

    # 当前播放头
    play_x = int(
        round(
            x0 + (x1 - x0) * int(frame_idx) / max(1, timeline["frame_count"] - 1)
        )
    )

    cv2.line(
        panel,
        (play_x, 4),
        (play_x, panel_h - 8),
        played_color,
        2,
        cv2.LINE_AA,
    )

    cv2.circle(
        panel,
        (play_x, y_top),
        5,
        played_color,
        -1,
        cv2.LINE_AA,
    )

    # 时间文字
    fps_float = max(1e-6, float(fps))
    cur_t = int(frame_idx) / fps_float
    total_t = timeline["frame_count"] / fps_float
    label = f"ENERGY  {_fmt_time(cur_t)} / {_fmt_time(total_t)}"

    cv2.putText(
        panel,
        label,
        (x0, panel_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(cfg.get("font_scale", 0.55)),
        text_color,
        1,
        cv2.LINE_AA,
    )

    return np.vstack([frame, panel])


# ============================================================
# Render Stage
# ============================================================

@register("render")
class RenderStage(Stage):
    name = "render"

    def run(self, ctx):
        rc = ctx.config.get("render", {})

        if rc.get("mode", "decode_render") == "stream_copy":
            self._stream_copy(ctx)
            return ctx

        src_w = int(ctx.meta["width"])
        src_h = int(ctx.meta["height"])

        # 这里保留从 ctx.config 读取 output 的能力。
        # 如果 ctx.config 没有 output，就使用本文件顶部写死的 1280x720。
        output_cfg = ctx.config.get("output", {})
        out_w = int(output_cfg.get("width", DEFAULT_OUTPUT_WIDTH))
        out_h = int(output_cfg.get("height", DEFAULT_OUTPUT_HEIGHT))

        # fps 不写死，继续使用统一时间轴。
        # 你当前视频是 30fps，这里会拿到 30。
        fps = ctx.timeline.fps

        emax = effective_max_zoom(ctx.config, src_w, src_h, out_w, out_h)
        track = ctx.camera_track

        outputs = rc.get("outputs", DEFAULT_RENDER_OUTPUTS)
        paths = _out_paths(ctx.output_path)
        # 标注所需：分镜表/事件/拍点/C位/预C位
        shot_plan = ctx.extras.get("shot_plan") or []
        pose_events = (ctx.extras.get("pose") or {}).get("pose_events") or []
        beat_frames = [int(b["frame"]) for b in
                       ((ctx.extras.get("music") or {}).get("beat_grid") or [])]
        primary_records = ctx.extras.get("primary_records") or []
        backup_subjects = ctx.extras.get("backup_subjects") or []
        n_frames = ctx.timeline.frame_count
        if not shot_plan:
            outputs = [o for o in outputs if o != "debug"]   # 没分镜就没什么可标的

        audio = ctx.input_path if ctx.meta.get("has_audio") else None

        blur_cfg = DEFAULT_MOTION_BLUR_CONFIG.copy()
        blur_cfg.update(rc.get("motion_blur", {}))

        skcfg = DEFAULT_SKELETON_CONFIG.copy()
        skcfg.update(rc.get("skeleton", {}))

        lw = int(skcfg.get("line_width", 4))
        pr = int(skcfg.get("point_radius", 5))
        mc = float(skcfg.get("min_confidence", 0.2))

        # final 专用：底部追加音乐能量时间轴
        music_timeline = _prepare_music_timeline(ctx, out_w, out_h) if "final" in outputs else None
        final_h = out_h + (music_timeline["height"] if music_timeline else 0)

        if music_timeline:
            print(
                f"[render][timeline] enabled："
                f"主画面={out_w}x{out_h}，"
                f"时间轴={out_w}x{music_timeline['height']}，"
                f"final={out_w}x{final_h}，"
                f"fps={float(fps):.3f}"
            )

        # 骨架输出所需的主体逐帧关键点
        need_sk = ("skeleton_raw" in outputs) or ("skeleton_cam" in outputs)
        prim_by_f = {}

        if need_sk:
            prim = ctx.extras.get("primary_records") or self._load_primary(ctx)
            prim_by_f = {
                int(r["frame_index"]): r.get("primary_person")
                for r in prim
                if r.get("frame_index") is not None
            }

        writers = {}

        if "final" in outputs:
            writers["final"] = FrameWriter(
                paths["final"],
                out_w,
                final_h,
                fps,
                audio_from=audio,
                render_cfg=rc,
            )

        if "debug" in outputs:
            writers["debug"] = FrameWriter(
                paths.get("debug", paths["final"].replace(".mp4", "_debug.mp4")),
                out_w, final_h, fps, audio_from=audio, render_cfg=rc)

        if "skeleton_cam" in outputs:
            writers["skeleton_cam"] = FrameWriter(
                paths["skeleton_cam"],
                out_w,
                out_h,
                fps,
                audio_from=audio,
                render_cfg=rc,
            )

        if "skeleton_raw" in outputs:
            writers["skeleton_raw"] = FrameWriter(
                paths["skeleton_raw"],
                src_w,
                src_h,
                fps,
                audio_from=audio,
                render_cfg=rc,
            )

        prev_center = None
        written = 0
        _prog = Progress(ctx.timeline.frame_count, "渲染", every_frac=0.05, min_step=30)

        for i, frame in enumerate(frame_reader(ctx.input_path, src_w, src_h)):
            _prog.update(i)
            cam = track[i] if i < len(track) else track[-1]

            M, rect = camera_matrix(
                cam,
                src_w,
                src_h,
                out_w,
                out_h,
                emax,
            )

            person = prim_by_f.get(i) if need_sk else None

            # final / skeleton_cam 共用同一张运镜画面
            cam_img = None

            if "final" in outputs or "skeleton_cam" in outputs:
                cam_img = apply_to_frame(frame, M, out_w, out_h)

                if blur_cfg.get("enabled", True):
                    cx = rect[0] + rect[2] / 2.0
                    cy = rect[1] + rect[3] / 2.0

                    if prev_center is not None:
                        dx = cx - prev_center[0]
                        dy = cy - prev_center[1]
                    else:
                        dx = dy = 0.0

                    strength = max(
                        float(getattr(cam, "blur", 0.0)),
                        1.0
                        if math.hypot(dx, dy) >= float(blur_cfg.get("threshold_px", 14.0))
                        else 0.0,
                    )

                    if strength > 0:
                        cam_img = _directional_blur(cam_img, dx, dy, strength)

                    prev_center = (cx, cy)

            if "final" in outputs:
                # cam_img 是 1280x720 主画面；下面追加时间轴 → 1280x848
                # ★字幕：景别种类 + 该景别内的运镜动作；时间轴：换景别时刻(白竖线+三角)
                ann.draw_subtitle(cam_img, shot_plan, i, fps, zoom=cam.zoom)
                strip = ann.draw_timeline(cam_img.shape[1], 128, shot_plan, pose_events,
                                          beat_frames, n_frames, i, fps)
                writers["final"].write(np.vstack([cam_img, strip]))

            if "debug" in outputs:
                # ★调试版：C位框 + 预C位框(B1/B2) + 同样的字幕与时间轴。
                #   坐标要用与画面同一个仿射矩阵 M 变换过去，否则必然错位。
                dbg = cam_img.copy()
                # ★primary_records[i] 是包装记录，真正的人在 ["primary_person"] 里，
                #   直接对它取 box_xyxy 会拿到 None（框就画不出来）。
                _rec = primary_records[i] if i < len(primary_records) else None
                _person = (_rec or {}).get("primary_person")
                # ★从成片反量「实际拍成的景别」，与 shot_plan 的设计意图并排显示。
                #   两者不符 = camera 被 max_zoom/安全框夹住了 —— 这是最该看见的信息。
                _act = ann.achieved_shot(_person, M, cam_img.shape[1], cam_img.shape[0])
                _box = ann.draw_subtitle(dbg, shot_plan, i, fps, zoom=cam.zoom,
                                         extra="C=main subject  B1/B2=backup",
                                         actual=_act)
                ann.draw_subject_debug(dbg, _person,
                                       (backup_subjects[i] if i < len(backup_subjects) else None),
                                       M=M, avoid=_box)
                strip = ann.draw_timeline(dbg.shape[1], 128, shot_plan, pose_events,
                                          beat_frames, n_frames, i, fps)
                writers["debug"].write(np.vstack([dbg, strip]))

            if "skeleton_cam" in outputs:
                img = cam_img.copy()

                if person:
                    sk.draw_person_transformed(
                        img,
                        person,
                        M,
                        min_conf=mc,
                        line_width=lw,
                        point_radius=pr,
                    )

                writers["skeleton_cam"].write(img)

            if "skeleton_raw" in outputs:
                raw = frame.copy()

                if skcfg.get("background", "video") == "black":
                    raw[:, :] = 0

                if person:
                    sk.draw_person(
                        raw,
                        person,
                        min_conf=mc,
                        line_width=lw,
                        point_radius=pr,
                    )

                writers["skeleton_raw"].write(raw)

            written += 1

        for w in writers.values():
            w.close()

        made = ", ".join(
            f"{k}->{os.path.basename(paths[k])}"
            for k in writers
        )

        print(
            f"[render] decode_render："
            f"max_zoom={emax:.2f}，"
            f"写出 {written} 帧 · {made}"
        )

        if written != ctx.timeline.frame_count:
            print(
                f"[render][告警] 写出 {written} != 输入 {ctx.timeline.frame_count}"
                f"（可能丢帧）"
            )

        return ctx

    def _load_primary(self, ctx):
        p = ctx.extras.get("primary_keypoints_path")

        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)

        return []

    def _stream_copy(self, ctx):
        import subprocess

        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            ctx.input_path,
            "-c",
            "copy",
            ctx.output_path,
        ]

        if subprocess.run(cmd).returncode != 0:
            raise RuntimeError("stream_copy 失败")

        print(f"[render] stream_copy -> {ctx.output_path}")
