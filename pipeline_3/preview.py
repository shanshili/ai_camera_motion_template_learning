"""
模板预览渲染（库模块，供 learn_template 流程调用；非独立脚本）
=============================================================
合成白底火柴人（编排 跳/伸展/旋转/下蹲/定格/大位移）+ 合成节拍段落，
用真实 shotplan + camera + 模板链路算运镜，渲成预览视频，底部带动作时间轴。

对外只暴露 render_preview(...)；模板学习完成后自动调它，直观看学到的模板怎么运镜。
按模板朝向（landscape/portrait/square）选择源/输出画布，保证预览与套用朝向一致。
"""

import os

import cv2
import numpy as np

from .context import PipelineContext
from .timeline import Timeline
from . import template as tmpl
from . import subject as subj
from . import annotate as ann
from . import yolo_pose as pose_utils
from . import pose_events as pe
from . import skeleton as sk
from .transform import camera_matrix, effective_max_zoom
from .stages.shotplan import ShotPlanStage
from .stages.camera import CameraStage
from .log import log, Progress


# 朝向 → (源画布, 输出画面)
_CANVAS = {
    "portrait":  ((1080, 1920), (720, 1280)),
    "landscape": ((1920, 1080), (1280, 720)),
    "square":    ((1080, 1080), (720, 720)),
}

_BASE = {
    "nose": (0, -95), "left_eye": (-8, -100), "right_eye": (8, -100),
    "left_ear": (-16, -98), "right_ear": (16, -98),
    "left_shoulder": (-40, -60), "right_shoulder": (40, -60),
    "left_elbow": (-60, -20), "right_elbow": (60, -20),
    "left_wrist": (-70, 10), "right_wrist": (70, 10),
    "left_hip": (-25, 20), "right_hip": (25, 20),
    "left_knee": (-24, 70), "right_knee": (24, 70),
    "left_ankle": (-22, 120), "right_ankle": (22, 120),
}
_SHOULDER_ARM = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist"}
_LIMB_ENDS = {"left_wrist", "right_wrist", "left_ankle", "right_ankle"}


def _make_person(cx, cy, s, sw_mul=1.0, spread_mul=1.0):
    kps, xs, ys = [], [], []
    for name, (dx, dy) in _BASE.items():
        fx = sw_mul if name in _SHOULDER_ARM else 1.0
        fs = spread_mul if name in _LIMB_ENDS else 1.0
        x = cx + dx * s * fx * fs
        y = cy + dy * s * fs
        kps.append({"name": name, "xy": [x, y], "xyn": None, "confidence": 0.95})
        xs.append(x); ys.append(y)
    return {"person_index": 0, "tracker_id": 1,
            "box_xyxy": [min(xs), min(ys), max(xs), max(ys)],
            "score": 0.95, "keypoints": kps}


def _choreograph(n, fps, src_w, src_h):
    cx0, cy0, s = src_w * 0.5, src_h * 0.55, min(src_w, src_h) / 1080 * 2.8
    body_h = 215 * s
    records = []
    at = lambda sec: int(round(sec * fps))
    for i in range(n):
        t = i / fps
        cx = cx0 + 0.06 * src_w * np.sin(t * 1.2)
        cy, sw_mul, spread_mul = cy0, 1.0, 1.0
        for jt in (2.5, 16.0):
            if at(jt) <= i < at(jt) + int(0.35 * fps):
                cy -= 0.16 * body_h
        if at(5.0) <= i < at(5.0) + int(0.5 * fps):
            spread_mul = 1.7
        if at(8.0) <= i < at(8.0) + int(0.5 * fps):
            sw_mul = 0.35
        if at(10.5) <= i < at(10.5) + int(1.5 * fps):
            cy += 0.22 * body_h
        if at(13.0) <= i < at(13.0) + int(1.2 * fps):
            cx = cx0
        if at(18.5) <= i < at(18.5) + int(1.2 * fps):
            cx = cx0 + 0.28 * src_w * ((i - at(18.5)) / (1.2 * fps))
        records.append({"frame_index": i, "original_shape": [src_h, src_w],
                        "people": [_make_person(cx, cy, s, sw_mul, spread_mul)]})
    return records


def _synth_music(n, fps, bpm):
    period = 60.0 / max(1.0, bpm)
    beat_grid, k, t = [], 0, 0.0
    while t < n / fps:
        f = int(round(t * fps))
        if 0 <= f < n:
            beat_grid.append({"t": round(t, 3), "frame": f, "is_downbeat": (k % 4 == 0),
                              "strength": round(0.5 + 0.4 * (k % 4 == 0), 3)})
        k += 1; t += period
    third = max(1, n // 3)
    sections = [{"start_f": 0, "end_f": third, "label": "low", "energy": 0.25},
                {"start_f": third, "end_f": 2 * third, "label": "mid", "energy": 0.55},
                {"start_f": 2 * third, "end_f": n, "label": "high", "energy": 0.85}]
    energy = np.clip(np.concatenate([np.full(third, 0.25), np.full(third, 0.55),
                                     np.full(n - 2 * third, 0.85)]), 0, 1)
    return {"fps": float(fps), "bpm": float(bpm), "n_frames": n, "beat_grid": beat_grid,
            "sections": sections, "energy_curve": [round(float(x), 4) for x in energy],
            "abs_loudness": 0.35}


_SHOT_COLOR = {"extreme_wide": (180, 150, 90), "wide": (110, 175, 95),
               "medium": (90, 150, 225), "closeup": (90, 90, 235)}
_EVENT_COLOR = {"jump": (60, 60, 235), "spin": (200, 120, 90), "extension": (90, 200, 90),
                "freeze": (150, 150, 150), "level_change": (60, 150, 235),
                "focus_switch": (200, 90, 200)}


def _draw_timeline(w, strip_h, plan, events, beats, n, fps, cur_i):
    panel = np.full((strip_h, w, 3), 22, np.uint8)
    x0, x1 = 40, w - 40
    yb0, yb1 = 14, strip_h - 40
    X = lambda f: int(x0 + (x1 - x0) * f / max(1, n - 1))
    for seg in plan:
        cv2.rectangle(panel, (X(seg["start_f"]), yb0), (X(seg["end_f"]), yb1),
                      _SHOT_COLOR.get(seg["shot"], (120, 120, 120)), -1)
    for seg in plan[1:]:
        cv2.line(panel, (X(seg["start_f"]), yb0 - 4), (X(seg["start_f"]), yb1 + 4),
                 (255, 255, 255), 1, cv2.LINE_AA)
    for b in beats:
        bx = X(int(b["frame"]))
        cv2.line(panel, (bx, yb1 + 6), (bx, yb1 + 14),
                 (80, 80, 240) if b.get("is_downbeat") else (90, 90, 90), 1, cv2.LINE_AA)
    for ev in events:
        ex = X(int(ev["frame"])); col = _EVENT_COLOR.get(ev.get("type"), (200, 200, 200))
        cv2.drawMarker(panel, (ex, yb0 - 6), col, cv2.MARKER_TRIANGLE_DOWN, 12, 2)
        cv2.putText(panel, ev.get("type", "")[:4], (ex - 12, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
    px = X(cur_i)
    cv2.line(panel, (px, 2), (px, strip_h - 2), (60, 220, 240), 2, cv2.LINE_AA)
    cv2.putText(panel, f"{cur_i/fps:5.2f}s / {n/fps:5.2f}s", (x0, strip_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return panel


def _build_ctx(records, music, template_path, out_dir, src_w, src_h, out_w, out_h, fps):
    primary, meta = subj.select_subject(records, src_w, src_h, fps,
                                        sections=music.get('sections'))
    primary_records = pose_utils.build_primary_records(records, primary)
    for r, mode, gb in zip(primary_records, meta["compose_mode"], meta["group_box"]):
        r["compose_mode"] = mode; r["group_box"] = gb
    series = pose_utils.primary_series_to_kpts(primary)
    pose = pe.analyze_pose_from_kpts(series, fps, jmap=pe.COCO17_MAP)
    for f in meta["focus_switch_frames"]:
        pose["pose_events"].append({"frame": int(f), "type": "focus_switch", "intensity": 0.5})
    pose["pose_events"].sort(key=lambda e: e["frame"])
    config = {
        "io": {"input": "<preview>", "output": os.path.join(out_dir, "preview.mp4")},
        "output": {"width": out_w, "height": out_h},
        "template": {"path": template_path},
        "shotplan": {"enabled": True},
        "camera": {"type": "rule", "max_zoom": 1.45, "allow_upscale_for_demo": False,
                   "follow": {"min_zoom": 1.0, "min_cutoff": 0.45, "beta": 0.03,
                              "zoom_min_cutoff": 0.30, "zoom_beta": 0.015,
                              "deadzone_x": 0.10, "deadzone_y": 0.12, "max_center_step_px": 32.0},
                   "safe_frame": {"enabled": True, "margin_frac": 0.06, "core_margin_frac": 0.10,
                                  "core_center_pull": 0.6, "core_y_anchor": 0.42,
                                  "fullbody_events": ["level_change", "freeze", "floor"],
                                  "fullbody_pad_sec": 0.6,
                                  "downgrade_ratio": {"wide": 0.82, "medium": 0.68, "upper": 0.55}},
                   "rotation": {"enabled": True, "max_deg": 10}},
    }
    ctx = PipelineContext(config=config, input_path="<preview>",
                          output_path=config["io"]["output"])
    ctx.meta = {"width": src_w, "height": src_h, "has_audio": False}
    ctx.timeline = Timeline(fps=fps, frame_count=len(records))
    ctx.extras["analysis_out_dir"] = out_dir
    ctx.extras["music"] = music
    ctx.extras["pose"] = pose
    ctx.extras["primary_records"] = primary_records
    return ctx, primary, pose


def _open_writer(out_path, fps, size):
    """
    稳健打开 VideoWriter：依次尝试 mp4v(.mp4) → XVID(.avi) → MJPG(.avi)。
    Windows 的 OpenCV 常打不开 mp4v，回退到 avi 几乎必成。
    返回 (writer, 实际路径)；都失败返回 (None, None)。
    """
    base = os.path.splitext(out_path)[0]
    candidates = [("mp4v", out_path if out_path.lower().endswith(".mp4") else base + ".mp4"),
                  ("XVID", base + ".avi"),
                  ("MJPG", base + ".avi")]
    for fourcc_str, path in candidates:
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_str), fps, size)
        if w.isOpened():
            if fourcc_str != "mp4v":
                log(f"[预览] mp4v 不可用，改用 {fourcc_str} -> {os.path.basename(path)}")
            return w, path
        w.release()
    return None, None


def _truncate(records, music, m):
    """把 records/music 截到前 m 帧（用于 --preview-seconds 上限；None 则不截）。"""
    if m is None or m >= len(records):
        return records, music
    records = records[:m]
    mu = dict(music)
    mu["beat_grid"] = [b for b in music.get("beat_grid", []) if int(b.get("frame", 0)) < m]
    mu["sections"] = [{**s, "end_f": min(int(s["end_f"]), m)}
                      for s in music.get("sections", []) if int(s["start_f"]) < m]
    ec = music.get("energy_curve", [])
    mu["energy_curve"] = ec[:m] if ec else ec
    return records, mu


def render_preview_from_data(template_path, out_path, records, music,
                             orientation="portrait", fps=30, max_seconds=None):
    """
    用**参考视频自己的真实骨架 + 真实音乐**渲染预览：把学到的模板套上去。
    长度 = 输入视频长度（除非 max_seconds 设了上限）。
    画面：全体骨架淡显 + C位加粗，过模板算出的相机；底部动作时间轴。
    这才是"这个模板套到这条视频会怎么运镜"的真实预览，故与输入等长。
    """
    try:
        records, music = _truncate(records, music, max_seconds and int(round(max_seconds * fps)))
        n = len(records)
        if n == 0:
            log("[预览][告警] 无 records，跳过预览")
            return None
        sh = records[0].get("original_shape") or [1920, 1080]
        src_h, src_w = int(sh[0]), int(sh[1])
        (_, _), (out_w, out_h) = _CANVAS.get(orientation, _CANVAS["portrait"])
        fps = int(round(fps)) or 30

        out_path = os.path.abspath(out_path)
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        log(f"预览：目标文件 {out_path}")
        log(f"预览：真实数据驱动，{n} 帧（≈{n/fps:.1f}s，与输入等长）· 源{src_w}x{src_h} 出{out_w}x{out_h}")

        ctx, primary, pose = _build_ctx(records, music, template_path, out_dir,
                                        src_w, src_h, out_w, out_h, fps)
        ctx = ShotPlanStage().run(ctx)
        ctx = CameraStage().run(ctx)
        plan, track, events = ctx.extras["shot_plan"], ctx.camera_track, pose["pose_events"]

        emax = effective_max_zoom(ctx.config, src_w, src_h, out_w, out_h)
        strip_h = 130
        writer, real_path = _open_writer(out_path, fps, (out_w, out_h + strip_h))
        if writer is None:
            log("[预览][告警] mp4v/XVID/MJPG 都打不开 VideoWriter，跳过预览")
            return None

        faint = (150, 200, 170)   # 其他人：淡色
        prog = Progress(n, "预览", every_frac=0.1, min_step=60)
        for i in range(n):
            prog.update(i)
            cam = track[i] if i < len(track) else track[-1]
            M, _ = camera_matrix(cam, src_w, src_h, out_w, out_h, emax)
            canvas = np.full((out_h, out_w, 3), 255, np.uint8)
            # 先画全体（淡），再画 C位（加粗，默认亮色）
            for p in (records[i].get("people") or []):
                sk.draw_person_transformed(canvas, p, M, min_conf=0.2, line_width=2,
                                           point_radius=3, edge_color=faint, point_color=faint)
            if primary[i]:
                sk.draw_person_transformed(canvas, primary[i], M, min_conf=0.2,
                                           line_width=5, point_radius=6)
            ann.draw_subtitle(canvas, plan, i, fps, zoom=cam.zoom)
            strip = ann.draw_timeline(out_w, strip_h, plan, events,
                                      [b["frame"] for b in music.get("beat_grid", [])],
                                      n, i, fps)
            writer.write(np.vstack([canvas, strip]))
        writer.release()
        size = os.path.getsize(real_path) if os.path.exists(real_path) else 0
        if size < 1024:
            log(f"[预览][告警] 写出文件过小（{size}B），编码器可能未工作：{real_path}")
            return None
        log(f"预览完成 -> {real_path}（{size//1024}KB, {n}帧, 分镜 {len(plan)} 段, 事件 {len(events)} 个）")
        return real_path
    except Exception as e:
        log(f"[预览][告警] 渲染失败（{type(e).__name__}: {e}），跳过预览")
        return None


def render_preview(template_path, out_path, orientation="portrait",
                   seconds=22.0, fps=30, bpm=120.0):
    """
    渲染一段该模板的火柴人运镜预览（供 learn_template 流程自动调用）。
    orientation ∈ {portrait, landscape, square}，决定画布朝向（与模板限定一致）。
    失败不抛出（预览是附带产物），只打印告警并返回 None。
    返回实际写出的文件路径（编码器回退时扩展名可能变 .avi）。
    """
    try:
        (src_w, src_h), (out_w, out_h) = _CANVAS.get(orientation, _CANVAS["portrait"])
        fps = int(fps)
        n = int(round(seconds * fps))
        out_path = os.path.abspath(out_path)
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        log(f"预览：目标文件 {out_path}")
        log(f"预览：合成 {n} 帧（{orientation} {out_w}x{out_h}, BPM={bpm}）…")
        records = _choreograph(n, fps, src_w, src_h)
        music = _synth_music(n, fps, bpm)
        ctx, primary, pose = _build_ctx(records, music, template_path, out_dir,
                                        src_w, src_h, out_w, out_h, fps)
        ctx = ShotPlanStage().run(ctx)
        ctx = CameraStage().run(ctx)
        plan, track, events = ctx.extras["shot_plan"], ctx.camera_track, pose["pose_events"]

        emax = effective_max_zoom(ctx.config, src_w, src_h, out_w, out_h)
        strip_h = 130
        writer, real_path = _open_writer(out_path, fps, (out_w, out_h + strip_h))
        if writer is None:
            log("[预览][告警] mp4v/XVID/MJPG 都打不开 VideoWriter，跳过预览渲染。"
                "（可 pip 安装带编码器的 opencv，或改输出到 .avi）")
            return None
        prog = Progress(n, "预览", every_frac=0.2, min_step=30)
        for i in range(n):
            prog.update(i)
            cam = track[i] if i < len(track) else track[-1]
            M, _ = camera_matrix(cam, src_w, src_h, out_w, out_h, emax)
            canvas = np.full((out_h, out_w, 3), 255, np.uint8)
            sk.draw_person_transformed(canvas, primary[i], M, min_conf=0.2,
                                       line_width=5, point_radius=6)
            ann.draw_subtitle(canvas, plan, i, fps, zoom=cam.zoom)
            strip = ann.draw_timeline(out_w, strip_h, plan, events,
                                      [b["frame"] for b in music["beat_grid"]],
                                      n, i, fps)
            writer.write(np.vstack([canvas, strip]))
        writer.release()
        size = os.path.getsize(real_path) if os.path.exists(real_path) else 0
        if size < 1024:
            log(f"[预览][告警] 写出的文件过小（{size}B），编码器可能没真正工作：{real_path}")
            return None
        log(f"预览完成 -> {real_path}（{size//1024}KB, 分镜 {len(plan)} 段, 事件 {len(events)} 个）")
        return real_path
    except Exception as e:
        log(f"[预览][告警] 渲染失败（{type(e).__name__}: {e}），跳过预览")
        return None
