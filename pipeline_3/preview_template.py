#!/usr/bin/env python3
"""
模板可视化预览（离线，独立运行）
================================
不需要真实视频 / YOLO：合成一个白底火柴人（编排好 跳/伸展/旋转/下蹲/定格 等动作）
+ 合成节拍/段落/能量，然后用**真实的 shotplan + camera + 模板**链路算出运镜，
渲成预览视频；底部画一条"动作时间轴"，标出换镜边界、逐段景别、姿态事件、拍点。

用途：直观检查一个 template.json 到底会怎么运镜，而不必先有真实素材。

用法：
  python -m pipeline_3.preview_template --out preview.mp4                 # 用内置默认模板
  python -m pipeline_3.preview_template --template templates/x.json --out preview.mp4
  python -m pipeline_3.preview_template --out p.mp4 --seconds 24 --bpm 128
"""

import argparse
import os

import cv2
import numpy as np

from .context import PipelineContext, CameraParams
from .timeline import Timeline
from . import template as tmpl
from . import subject as subj
from . import yolo_pose as pose_utils
from . import pose_events as pe
from . import skeleton as sk
from .transform import camera_matrix, effective_max_zoom
from .stages.shotplan import ShotPlanStage
from .stages.camera import CameraStage
from .log import log, Progress


# COCO-17 长名基准骨架（相对质心的偏移，单位后续乘 scale）
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


def make_person(cx, cy, s, sw_mul=1.0, spread_mul=1.0):
    """生成一帧火柴人 person（长名关键点，源坐标）。
    sw_mul<1 收窄双肩（模拟旋转 spin）；spread_mul>1 肢体外张（模拟伸展 extension）。"""
    kps = []
    xs, ys = [], []
    for name, (dx, dy) in _BASE.items():
        fx = sw_mul if name in _SHOULDER_ARM else 1.0
        fs = spread_mul if name in _LIMB_ENDS else 1.0
        x = cx + dx * s * fx * fs
        y = cy + dy * s * fs
        kps.append({"name": name, "xy": [x, y], "xyn": None, "confidence": 0.95})
        xs.append(x); ys.append(y)
    box = [min(xs), min(ys), max(xs), max(ys)]
    return {"person_index": 0, "tracker_id": 1, "box_xyxy": box,
            "score": 0.95, "keypoints": kps}


def choreograph(n, fps, src_w, src_h):
    """编排一段带明显事件的独舞，返回逐帧 records（单人）。"""
    cx0, cy0, s = src_w * 0.5, src_h * 0.55, 2.8
    body_h = 215 * s
    records = []

    def at(sec):
        return int(round(sec * fps))

    # 事件时间表（秒）：类型 + 持续
    for i in range(n):
        t = i / fps
        cx = cx0 + 0.06 * src_w * np.sin(t * 1.2)     # 轻微左右摇摆
        cy = cy0
        sw_mul, spread_mul = 1.0, 1.0

        # 跳：全身上移
        for jt in (2.5, 16.0):
            if at(jt) <= i < at(jt) + int(0.35 * fps):
                cy -= 0.16 * body_h
        # 伸展：肢体外张
        if at(5.0) <= i < at(5.0) + int(0.5 * fps):
            spread_mul = 1.7
        # 旋转：收窄双肩（持续 ~2 拍）
        if at(8.0) <= i < at(8.0) + int(0.5 * fps):
            sw_mul = 0.35
        # 下蹲/重心变化：质心持续下移
        if at(10.5) <= i < at(10.5) + int(1.5 * fps):
            cy += 0.22 * body_h
        # 定格：完全静止（覆盖摇摆）
        if at(13.0) <= i < at(13.0) + int(1.2 * fps):
            cx = cx0
        # 大位移旅行：横向大幅移动
        if at(18.5) <= i < at(18.5) + int(1.2 * fps):
            u = (i - at(18.5)) / (1.2 * fps)
            cx = cx0 + (0.28 * src_w) * u

        person = make_person(cx, cy, s, sw_mul=sw_mul, spread_mul=spread_mul)
        records.append({"frame_index": i, "original_shape": [src_h, src_w],
                        "people": [person]})
    return records


def synth_music(n, fps, bpm):
    """合成节拍/段落/能量：整段分三段 low/mid/high，能量随之抬升。"""
    period = 60.0 / bpm
    beat_grid = []
    k = 0
    t = 0.0
    while t < n / fps:
        f = int(round(t * fps))
        if 0 <= f < n:
            beat_grid.append({"t": round(t, 3), "frame": f,
                              "is_downbeat": (k % 4 == 0),
                              "strength": round(0.5 + 0.4 * (k % 4 == 0), 3)})
        k += 1
        t += period
    third = n // 3
    sections = [
        {"start_f": 0, "end_f": third, "label": "low", "energy": 0.25},
        {"start_f": third, "end_f": 2 * third, "label": "mid", "energy": 0.55},
        {"start_f": 2 * third, "end_f": n, "label": "high", "energy": 0.85},
    ]
    energy = np.concatenate([
        np.full(third, 0.25), np.full(third, 0.55), np.full(n - 2 * third, 0.85)])
    energy = energy + 0.08 * np.sin(np.arange(n) * 2 * np.pi / max(1, int(fps * period)))
    energy = np.clip(energy, 0, 1)
    return {"fps": float(fps), "bpm": float(bpm), "n_frames": n,
            "beat_grid": beat_grid, "sections": sections,
            "energy_curve": [round(float(x), 4) for x in energy],
            "abs_loudness": 0.35}


# ----------------------------- 时间轴绘制 -----------------------------
_SHOT_COLOR = {  # BGR
    "extreme_wide": (180, 150, 90), "wide": (110, 175, 95),
    "medium": (90, 150, 225), "closeup": (90, 90, 235),
}
_EVENT_COLOR = {
    "jump": (60, 60, 235), "spin": (200, 120, 90), "extension": (90, 200, 90),
    "freeze": (150, 150, 150), "level_change": (60, 150, 235),
    "focus_switch": (200, 90, 200),
}


def draw_timeline(w, strip_h, plan, events, beats, n, fps, cur_i):
    panel = np.full((strip_h, w, 3), 22, np.uint8)
    x0, x1 = 40, w - 40
    y_band0, y_band1 = 14, strip_h - 40

    def X(f):
        return int(x0 + (x1 - x0) * f / max(1, n - 1))

    # 逐段景别色带
    for seg in plan:
        c = _SHOT_COLOR.get(seg["shot"], (120, 120, 120))
        cv2.rectangle(panel, (X(seg["start_f"]), y_band0), (X(seg["end_f"]), y_band1), c, -1)
    # 换镜边界（白竖线）
    for seg in plan[1:]:
        cv2.line(panel, (X(seg["start_f"]), y_band0 - 4), (X(seg["start_f"]), y_band1 + 4),
                 (255, 255, 255), 1, cv2.LINE_AA)
    # 拍点/强拍（底部细线）
    for b in beats:
        bx = X(int(b["frame"]))
        col = (80, 80, 240) if b.get("is_downbeat") else (90, 90, 90)
        cv2.line(panel, (bx, y_band1 + 6), (bx, y_band1 + 14), col, 1, cv2.LINE_AA)
    # 姿态事件（彩色三角 + 类型）
    for ev in events:
        ex = X(int(ev["frame"]))
        col = _EVENT_COLOR.get(ev.get("type"), (200, 200, 200))
        cv2.drawMarker(panel, (ex, y_band0 - 6), col, cv2.MARKER_TRIANGLE_DOWN, 12, 2)
        cv2.putText(panel, ev.get("type", "")[:4], (ex - 12, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
    # 播放头
    px = X(cur_i)
    cv2.line(panel, (px, 2), (px, strip_h - 2), (60, 220, 240), 2, cv2.LINE_AA)
    # 图例/时间
    t = cur_i / fps
    cv2.putText(panel, f"{t:5.2f}s / {n/fps:5.2f}s", (x0, strip_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return panel


# ----------------------------- 主流程 -----------------------------
def build_ctx(records, music, template_path, out_dir, src_w, src_h, out_w, out_h, fps):
    n = len(records)
    # 主体选择（单人 → 平凡 C位）+ 事件
    primary, meta = subj.select_subject(records, src_w, src_h, fps)
    primary_records = pose_utils.build_primary_records(records, primary)
    for r, mode, gb in zip(primary_records, meta["compose_mode"], meta["group_box"]):
        r["compose_mode"] = mode; r["group_box"] = gb
    series = pose_utils.primary_series_to_kpts(primary)
    pose = pe.analyze_pose_from_kpts(series, fps, jmap=pe.COCO17_MAP)
    for f in meta["focus_switch_frames"]:
        pose["pose_events"].append({"frame": int(f), "type": "focus_switch", "intensity": 0.5})
    pose["pose_events"].sort(key=lambda e: e["frame"])

    config = {
        "io": {"input": "<synthetic>", "output": os.path.join(out_dir, "preview.mp4")},
        "output": {"width": out_w, "height": out_h},
        "template": {"path": template_path},
        "shotplan": {"enabled": True},
        "camera": {
            "type": "rule", "max_zoom": 1.45, "allow_upscale_for_demo": False,
            "follow": {"min_zoom": 1.0, "min_cutoff": 0.45, "beta": 0.03,
                       "zoom_min_cutoff": 0.30, "zoom_beta": 0.015,
                       "deadzone_x": 0.10, "deadzone_y": 0.12, "max_center_step_px": 32.0},
            "safe_frame": {"enabled": True, "margin_frac": 0.06, "core_margin_frac": 0.10,
                           "core_center_pull": 0.6, "core_y_anchor": 0.42,
                           "fullbody_events": ["level_change", "freeze", "floor"],
                           "fullbody_pad_sec": 0.6,
                           "downgrade_ratio": {"wide": 0.82, "medium": 0.68, "upper": 0.55}},
            "rotation": {"enabled": True, "max_deg": 10},
        },
    }
    ctx = PipelineContext(config=config, input_path="<synthetic>",
                          output_path=config["io"]["output"])
    ctx.meta = {"width": src_w, "height": src_h, "has_audio": False}
    ctx.timeline = Timeline(fps=fps, frame_count=n)
    ctx.extras["analysis_out_dir"] = out_dir
    ctx.extras["music"] = music
    ctx.extras["pose"] = pose
    ctx.extras["primary_records"] = primary_records
    return ctx, primary, pose


def main():
    ap = argparse.ArgumentParser(description="模板运镜可视化预览（火柴人 + 动作时间轴）")
    ap.add_argument("--template", default=None, help="template.json；缺省用内置默认模板")
    ap.add_argument("--out", default="preview.mp4")
    ap.add_argument("--seconds", type=float, default=22.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--src", default="1080x1920", help="源画布 WxH")
    ap.add_argument("--outsize", default="720x1280", help="输出画面 WxH")
    args = ap.parse_args()

    src_w, src_h = map(int, args.src.lower().split("x"))
    out_w, out_h = map(int, args.outsize.lower().split("x"))
    fps = int(args.fps)
    n = int(round(args.seconds * fps))
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    log(f"合成编排 {n} 帧 @ {fps}fps（{args.seconds}s, BPM={args.bpm}）源{src_w}x{src_h} 出{out_w}x{out_h}")
    records = choreograph(n, fps, src_w, src_h)
    music = synth_music(n, fps, args.bpm)

    # 若给了模板路径就用它，否则把默认模板落一个临时文件供 shotplan 读
    tpath = args.template
    if not tpath:
        tpath = os.path.join(out_dir, "_default_template.json")
        tmpl.save_template(tmpl.default_template(), tpath)

    ctx, primary, pose = build_ctx(records, music, tpath, out_dir,
                                   src_w, src_h, out_w, out_h, fps)

    log("运行 shotplan（模板驱动）…")
    ctx = ShotPlanStage().run(ctx)
    log("运行 camera（安全执行）…")
    ctx = CameraStage().run(ctx)
    plan = ctx.extras["shot_plan"]
    track = ctx.camera_track
    events = pose["pose_events"]

    # 渲染：白底火柴人过相机矩阵 + 底部动作时间轴
    emax = effective_max_zoom(ctx.config, src_w, src_h, out_w, out_h)
    strip_h = 130
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (out_w, out_h + strip_h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter 打不开，检查编解码器/输出路径")

    log("渲染预览…")
    prog = Progress(n, "预览", every_frac=0.1, min_step=15)
    for i in range(n):
        prog.update(i)
        cam = track[i] if i < len(track) else track[-1]
        M, _ = camera_matrix(cam, src_w, src_h, out_w, out_h, emax)
        canvas = np.full((out_h, out_w, 3), 255, np.uint8)   # 白底
        sk.draw_person_transformed(canvas, primary[i], M, min_conf=0.2,
                                   line_width=5, point_radius=6)
        # HUD：当前景别/运镜
        cv2.putText(canvas, f"{cam.shot} / {cam.move}  zoom={cam.zoom:.2f}",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        strip = draw_timeline(out_w, strip_h, plan, events, music["beat_grid"], n, fps, i)
        writer.write(np.vstack([canvas, strip]))
    writer.release()
    log(f"完成 -> {args.out}（{n} 帧, 分镜 {len(plan)} 段, 事件 {len(events)} 个）")


if __name__ == "__main__":
    main()
