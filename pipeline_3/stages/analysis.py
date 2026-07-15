"""
阶段 2：分析 (Analysis) · 多人版
================================
姿态：读多人 tracked_keypoints.json（由 multi_person_tracking 前端离线产出）
      -> subject.select_subject 选 C位（角色化，绑位置不绑 id）
      -> primary_records（与旧单人 schema 一致）+ compose_mode/group_box
      -> 主体 17 点 -> pose_events 出 pose.json（并注入 focus_switch 事件）
音乐：analyze_music -> music.json（拍点/能量/段落/abs_loudness）
产物落 analysis.out_dir，同时挂到 ctx.extras 供后续阶段直接用。

★与旧版差异：
  - 不再在管线内跑 YOLO / select_primary_track；多人检测+跟踪是独立前端。
  - 主体选择换成 subject（C位 + 迟滞 + FEATURE/GROUP）。
  - 若只有旧单人 keypoints.json，也能退化读取（people 无 tracker_id，subject 照样工作）。
"""

import json
import os
from collections import Counter
from pathlib import Path

from ..stage import Stage, register
from ..log import log
from .. import music as music_mod
from .. import yolo_pose as pose_utils
from .. import subject as subj_mod
from .. import pose_events as pe


def _norm(path, base="."):
    path = str(path).replace("\\", os.sep)
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base, path))


def _strip_debug(d):
    return {k: v for k, v in d.items() if k != "_debug"}


def _align_len(series, n):
    if len(series) > n:
        return series[:n]
    return series + [None] * (n - len(series))


@register("analysis")
class AnalysisStage(Stage):
    name = "analysis"

    def run(self, ctx):
        cfg = ctx.config.get("analysis", {})
        if not cfg.get("enabled", True):
            print("[analysis] 已关闭：跳过 WP2/WP3")
            return ctx

        base_out = _norm(cfg.get("out_dir", "analysis_out"))
        if cfg.get("out_dir_per_video", True):
            out_dir = os.path.join(base_out, Path(str(ctx.input_path)).stem)
        else:
            out_dir = base_out
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ctx.extras["analysis_out_dir"] = out_dir

        fps = float(ctx.timeline.fps)
        n_frames = ctx.timeline.frame_count
        reuse = bool(cfg.get("reuse_existing", False))

        self._run_music(ctx, cfg, out_dir, fps, n_frames, reuse)
        self._run_pose(ctx, cfg, out_dir, fps, n_frames, reuse)
        return ctx

    # ------------------------- 音乐 -------------------------
    def _run_music(self, ctx, cfg, out_dir, fps, n_frames, reuse):
        mcfg = cfg.get("music", {})
        if not mcfg.get("enabled", True):
            return
        music_json = os.path.join(out_dir, "music.json")
        if reuse and os.path.exists(music_json):
            with open(music_json, encoding="utf-8") as f:
                music = json.load(f)
            print(f"[analysis.music] 复用 {music_json}")
        elif ctx.meta and ctx.meta.get("has_audio", False):
            log("音乐分析中：解码音频 → STFT/起音包络 → 速度/拍点 → 能量/分段 …")
            music = music_mod.analyze_music(ctx.input_path, fps, n_frames)
            music = _strip_debug(music)
            with open(music_json, "w", encoding="utf-8") as f:
                json.dump(music, f, ensure_ascii=False, indent=2)
            print(f"[analysis.music] BPM={music.get('bpm')} "
                  f"拍点={len(music.get('beat_grid', []))} "
                  f"段落={len(music.get('sections', []))} "
                  f"响度={music.get('abs_loudness')} -> {music_json}")
        else:
            music = {"fps": fps, "n_frames": n_frames, "beat_grid": [],
                     "energy_curve": [0.0] * n_frames, "abs_loudness": 0.0,
                     "sections": [{"start_f": 0, "end_f": n_frames,
                                   "label": "low", "energy": 0.0}]}
            with open(music_json, "w", encoding="utf-8") as f:
                json.dump(music, f, ensure_ascii=False, indent=2)
            print("[analysis.music][告警] 输入无音轨：生成空节拍/低能量结果")
        ctx.extras["music"] = music

    # ------------------------- 姿态 -------------------------
    def _run_pose(self, ctx, cfg, out_dir, fps, n_frames, reuse):
        pcfg = cfg.get("pose", {})
        if not pcfg.get("enabled", True):
            return

        tracked_json = os.path.join(out_dir, "tracked_keypoints.json")
        kpts_json = os.path.join(out_dir, "keypoints.json")
        primary_json = os.path.join(out_dir, "primary_keypoints.json")
        pose_json = os.path.join(out_dir, "pose.json")

        # 1) 取多人 records（优先多人 tracked，其次旧单人 keypoints）
        src = None
        if os.path.exists(tracked_json):
            src = tracked_json
        elif os.path.exists(kpts_json):
            src = kpts_json
        if src is None:
            raise FileNotFoundError(
                f"缺 {tracked_json}（或 {kpts_json}）。\n"
                "请先用 multi_person_tracking 前端对该视频跑一遍多人跟踪，"
                "产出 tracked_keypoints.json 放到该分析目录。")
        with open(src, encoding="utf-8") as f:
            records = json.load(f)
        print(f"[analysis.pose] 读 {os.path.basename(src)}（{len(records)} 帧）")

        # 2) 帧数对齐（守无漂移）
        if len(records) != n_frames:
            print(f"[analysis.pose][告警] records 帧数 {len(records)} != 视频帧数 {n_frames}，按 timeline 对齐")
            if len(records) > n_frames:
                records = records[:n_frames]
            else:
                for i in range(len(records), n_frames):
                    records.append({"frame_index": i, "original_shape":
                                    [ctx.meta["height"], ctx.meta["width"]], "people": []})

        # 3) 主体选择：C位角色化（多人/单人统一）
        #    传入音乐段落 → 启用段落锁定：一个唱段内不换 C位（治「C位乱聚焦」）
        music = ctx.extras.get("music", {}) or {}
        sections = music.get("sections") or None
        log(f"主体选择：C位角色化 + 段落锁定 + FEATURE/GROUP（{len(records)} 帧）…")
        primary_per_frame, sel_meta = subj_mod.select_subject(
            records, ctx.meta["width"], ctx.meta["height"], fps,
            cfg=pcfg.get("subject"), sections=sections)
        n_present = sum(1 for p in primary_per_frame if p)
        mode_cnt = Counter(sel_meta["compose_mode"])
        print(f"[analysis.pose] C位命中 {n_present}/{n_frames} 帧 · "
              f"模式{dict(mode_cnt)} · 换焦点{len(sel_meta['focus_switch_frames'])}次")

        primary_records = pose_utils.build_primary_records(records, primary_per_frame)
        # 附加多人信息（camera GROUP 路径 / shotplan 偏 wide 会用；缺则忽略）
        for r, mode, gbox in zip(primary_records, sel_meta["compose_mode"], sel_meta["group_box"]):
            r["compose_mode"] = mode
            r["group_box"] = gbox
        with open(primary_json, "w", encoding="utf-8") as f:
            json.dump(primary_records, f, ensure_ascii=False, indent=2)

        # 4) 主体 17 点 -> 事件机 -> pose.json（注入 focus_switch）
        kpts_series = pose_utils.primary_series_to_kpts(primary_per_frame)
        kpts_series = _align_len(kpts_series, n_frames)
        conf_th = float(pcfg.get("min_confidence", 0.3))
        pose = pe.analyze_pose_from_kpts(kpts_series, fps,
                                         jmap=pe.COCO17_MAP, conf_th=conf_th)
        for fnum in sel_meta["focus_switch_frames"]:
            pose["pose_events"].append({"frame": int(fnum), "type": "focus_switch",
                                        "intensity": 0.5})
        pose["pose_events"].sort(key=lambda e: e["frame"])
        with open(pose_json, "w", encoding="utf-8") as f:
            json.dump(pose, f, ensure_ascii=False, indent=2)
        cnt = Counter(e["type"] for e in pose.get("pose_events", []))
        print(f"[analysis.pose] 事件={len(pose.get('pose_events', []))} {dict(cnt)} -> {pose_json}")

        ctx.extras["pose"] = pose
        ctx.extras["primary_records"] = primary_records
        ctx.extras["backup_subjects"] = sel_meta.get("backup_subjects") or []
        ctx.extras["primary_keypoints_path"] = primary_json
        ctx.extras["keypoints_path"] = src
