#!/usr/bin/env python3
"""
模板学习模块（离线，独立运行）· 方案 §9
========================================
输入：一条「已剪辑的参考 MV」+ 它的分析产物（music.json / tracked_keypoints.json，
      由主管线 analysis 阶段先跑一遍参考视频得到）。
输出：template.json（生成端 shotplan/camera 直接读取）。

用法：
  # 先用主管线对参考视频跑一次 analysis（产出 music.json / tracked_keypoints.json）
  python learn_template.py \
      --ref-video refs/kpop_mv.mp4 \
      --analysis-dir analysis_out/kpop_mv \
      --out templates/kpop_mv_v1.json \
      --name kpop_mv_v1 --genre dance

诚实边界（对应 §9 / §12）：
  · 只反解「可复现的构图意图」：景别分布、换镜节奏、卡点命中率、段落→景别倾向。
  · 不做 homography 反解真实推轨/摇臂——裁剪相机复现不了露出画外内容的运动，
    强行学出来只会被 headroom/emax 夹死或产生错误主体尺度轨迹。
  · 相机运动仅由「主体在画面内的尺度轨迹」间接近似（push/pull），这是裁剪系统
    唯一能复现的成分。
  · 学不到的字段（强调形状、仲裁权重等）回落 default_template()。
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# 包内模块（运行时需能 import 到 pipeline 包；见文末 usage 说明）
from . import template as tmpl
from . import subject as subj
from . import yolo_pose as yp
from . import ffio


# ------------------------- 换镜检测（直方图相关，无额外依赖）-------------------------
def detect_cuts(video_path, width, height, sample_stride=1, mad_k=6.0,
                min_dist=0.15, min_gap_f=8):
    """
    硬切检测：相邻帧 HSV 直方图距离 d = 1 - corr 的**局部异常 + 绝对下限**。

    ★两个阈值缺一不可：
      · 只用固定阈值（旧写法 corr<0.6，即 d>0.4）：同场景切换（同舞台/同灯光/同服装）
        色彩直方图几乎不变，大面积漏检。
      · 只用自适应阈值（med + k·MAD）：极稳的片子里 med≈0.002、MAD≈0.002，
        阈值只有 0.012，会把灯光闪烁/压缩噪声全当成硬切（实测某片检出 60 个、实有 12 个）。
      → 取两者较大：既对"这条片子有多稳"自适应，又不低于真实切变的物理量级。
    min_dist：真实硬切的直方图距离至少该有多大（0.15 ≈ corr 0.85）。
    """
    if cv2 is None or not os.path.exists(video_path):
        print("[learn][告警] 无 cv2 或参考视频缺失，跳过换镜检测（cut_rhythm 用默认）")
        return []
    dists, idxs = [], []
    prev_hist = None
    for fi, frame in enumerate(ffio.frame_reader(video_path, width, height)):
        if fi % sample_stride:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            dists.append(1.0 - float(corr))
            idxs.append(fi)
        prev_hist = hist
    if not dists:
        return []
    d = np.asarray(dists)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) + 1e-9
    th = max(med + mad_k * mad, float(min_dist))
    cand = [(idxs[i], d[i]) for i in range(len(d)) if d[i] > th]
    cand.sort(key=lambda x: -x[1])
    cuts = []
    for f, _ in cand:
        if all(abs(f - c) >= min_gap_f for c in cuts):
            cuts.append(f)
    cuts.sort()
    top = np.sort(d)[-12:][::-1]
    print(f"[learn] 换镜检测：距离中位={med:.4f} MAD={mad:.4f} → 阈值={th:.4f}"
          f"（自适应 {med + mad_k*mad:.4f} vs 绝对下限 {min_dist}）→ {len(cuts)} 个硬切")
    print(f"[learn]   最大的 12 个帧间距离: {' '.join(f'{x:.3f}' for x in top)}"
          f"   ← 若与实际硬切数对不上，按这组数调 min_dist")
    return cuts


# ------------------------- 主体景别（按 C位露出/遮挡判定）-------------------------
def _kp_xy(person, names, conf_th=0.3):
    """取这些关节里第一个检测到的坐标（可能在画面外）。没检测到返回 None。"""
    for kp in (person.get("keypoints") or []):
        if kp.get("name") in names and kp.get("xy"):
            c = kp.get("confidence")
            if c is None or c >= conf_th:
                return kp["xy"]
    return None


def _shot_from_exposure(person, fw, fh, edge_pad=8):
    """
    按 C位「被画面裁掉多少」判景别 —— 这是"露出/遮挡"的正确读法。

    ★关键区分：「关节没检测到」≠「关节在画面外」。
      被别的舞者挡住 = 遮挡（漏检），画面其实是远景；
      被画幅切掉     = 取景意图（真的是中景/特写）。
      只有后者才是景别信号。所以先看 bbox 有没有贴到画面边缘：
        · 没贴边 → 整个人都在画面里 → 全身 → 按占画面大小分 wide / extreme_wide
        · 贴到底边 → 人被下边缘切了 → 再看最低的可见关节定切在哪：
            踝可见  → 仍近似全身      → wide
            膝可见  → 切在小腿        → wide
            髋可见  → 腰以上          → medium
            仅肩/头 → 胸以上          → closeup
    返回 (shot, exposure)；exposure ∈ full_body / half_body / head。
    """
    if not person:
        return None, None
    box = person.get("box_xyxy")
    if not (box and len(box) >= 4):
        return None, None
    y1 = float(box[3])
    r = max(0.0, y1 - float(box[1])) / max(1.0, fh)      # bbox 高占画面比

    # 没贴下边缘 → 全身在画面内（腿即便没检测到也是被挡住，不是被裁）
    if y1 < fh - edge_pad:
        return ("extreme_wide" if r < 0.45 else "wide"), "full_body"

    # 贴下边缘 → 被裁。看最低的可见关节切在哪一段
    ank = _kp_xy(person, ("left_ankle", "right_ankle"))
    knee = _kp_xy(person, ("left_knee", "right_knee"))
    hip = _kp_xy(person, ("left_hip", "right_hip"))
    inside = lambda p: p is not None and p[1] < fh - edge_pad

    if inside(ank):
        return ("extreme_wide" if r < 0.45 else "wide"), "full_body"
    if inside(knee):
        return "wide", "full_body"
    if inside(hip):
        return "medium", "half_body"
    return "closeup", "head"


# ------------------------- 段落切分 + 反解 -------------------------
def build_segments(cuts, n_frames):
    """换镜帧 → 段区间 [(s,e), ...]。无切则整片一段。"""
    bounds = [0] + [c for c in cuts if 0 < c < n_frames] + [n_frames]
    bounds = sorted(set(bounds))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]


def section_label_at(sections, f):
    for s in sections:
        if s["start_f"] <= f < s["end_f"]:
            return s.get("label", "mid")
    return "mid"


def learn(ref_video, analysis_dir, name, genre):
    # ---- 读骨架 records ----
    tk_path = os.path.join(analysis_dir, "tracked_keypoints.json")
    kp_path = os.path.join(analysis_dir, "keypoints.json")
    src = tk_path if os.path.exists(tk_path) else kp_path
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"{analysis_dir} 下没有 tracked_keypoints.json（或 keypoints.json）。\n"
            "  请先用 multi_person_tracking.run_tracking 对参考视频跑出骨架，"
            "且 --output-dir 就指向这个 --analysis-dir。")
    with open(src, encoding="utf-8") as f:
        records = json.load(f)
    n = len(records)

    # ---- fps：优先探测参考视频，失败退 30 ----
    fps = 30.0
    try:
        from . import ffio
        fps = float(ffio.ffprobe_meta(ref_video)["fps"])
    except Exception:
        pass

    # ---- music.json：有则读，无则现场分析（缺音轨/ffmpeg 则空音乐兜底）----
    music_json = os.path.join(analysis_dir, "music.json")
    if os.path.exists(music_json):
        with open(music_json, encoding="utf-8") as f:
            music = json.load(f)
        fps = float(music.get("fps", fps))
    else:
        try:
            from . import music as music_mod
            print("[learn] 未见 music.json，现场分析参考视频音乐…")
            music = {k: v for k, v in music_mod.analyze_music(ref_video, fps, n).items()
                     if k != "_debug"}
            with open(music_json, "w", encoding="utf-8") as f:
                json.dump(music, f, ensure_ascii=False, indent=2)
            print(f"[learn] 音乐分析完成 -> {music_json}")
        except Exception as e:
            print(f"[learn][告警] 音乐分析失败（{type(e).__name__}: {e}），用空音乐继续"
                  "（模板将不含节奏信息）")
            music = {"fps": fps, "bpm": None, "beat_grid": [],
                     "energy_curve": [0.0] * n, "abs_loudness": 0.0,
                     "sections": [{"start_f": 0, "end_f": n, "label": "mid", "energy": 0.0}]}

    fh = int(records[0].get("original_shape", [720, 1280])[0]) if n else 720
    fw = int(records[0].get("original_shape", [720, 1280])[1]) if n else 1280

    sections = music.get("sections") or [{"start_f": 0, "end_f": n, "label": "mid"}]
    beat_frames = np.array([int(b["frame"]) for b in music.get("beat_grid", [])], dtype=int)

    # ---- 反解：逐帧按 C位露出/遮挡定景别（不是 bbox 占比）+ 出镜程度 ----
    cuts = detect_cuts(ref_video, fw, fh)
    primary, sel_meta = subj.select_subject(records, fw, fh, fps,
                                            sections=sections, cut_frames=cuts)
    shot_dur = Counter()
    section_shot = defaultdict(Counter)
    exposure_cnt = Counter()          # C位出镜程度：full_body / half_body / head
    group_shot = Counter()            # ★群体(GROUP)帧上参考用过的景别
    for i, p in enumerate(primary):
        shot, exp = _shot_from_exposure(p, fw, fh)
        if shot is None:
            continue
        shot_dur[shot] += 1
        exposure_cnt[exp] += 1
        section_shot[section_label_at(sections, i)][shot] += 1
        if sel_meta["compose_mode"][i] == "GROUP":
            group_shot[shot] += 1

    total = sum(shot_dur.values()) or 1
    shot_hist = {k: round(shot_dur.get(k, 0) / total, 4) for k in tmpl.SHOT_LADDER}
    shot_bias = max(shot_hist, key=shot_hist.get) if total > 1 else "medium"
    exp_total = sum(exposure_cnt.values()) or 1
    c_exposure = {k: round(exposure_cnt.get(k, 0) / exp_total, 4)
                  for k in ("full_body", "half_body", "head")}

    # 段落→景别：★保留完整分布，不要只取众数。
    #   只取众数会把"20%特写+23%中景+41%远景"塌缩成"远景"，
    #   模板表达力退化成每个段落标签一个景别，MV 的景别变化全部丢失。
    base = tmpl.default_template()
    section_default = {k: dict(v) for k, v in base["section_default"].items()}
    section_shot_dist = {}
    for lab in ("low", "mid", "high"):
        c = section_shot[lab]
        if not c:
            continue
        tot = sum(c.values()) or 1
        section_shot_dist[lab] = {k: round(v / tot, 4) for k, v in c.most_common()}
        section_default[lab]["shot"] = c.most_common(1)[0][0]   # 仍留众数做兜底

    # ---- 换镜节奏 + 卡点命中率 ----
    seg_lens = [(e - s) for (s, e) in build_segments(cuts, n) if e > s]
    # ★夹住 cut_rhythm：换镜太少（原始未剪辑的固定机位视频）→ 段长≈全片，
    #   若直接采用会让 min_shot 大到把整片并成一段。少于 3 个切就用默认，
    #   否则把段长中位数夹到 [0.6, 4.0] 秒。
    if len(cuts) < 3:
        cut_rhythm_sec = float(base["style"]["cut_rhythm_sec"])
        print(f"[learn] 参考几乎无硬切（{len(cuts)}），cut_rhythm 用默认 {cut_rhythm_sec}s")
    else:
        cut_rhythm_sec = round(float(np.clip(np.median(seg_lens) / fps, 0.6, 4.0)), 3)

    beat_sync = 0.0
    if len(cuts) and len(beat_frames):
        win = max(1, int(round(0.15 * fps)))
        hit = sum(1 for c in cuts if np.min(np.abs(beat_frames - c)) <= win)
        beat_sync = round(hit / len(cuts), 4)

    # ---- 风格指纹 ----
    style_descriptor = [
        shot_hist["closeup"], shot_hist["medium"], shot_hist["wide"], shot_hist["extreme_wide"],
        cut_rhythm_sec, round(float(np.std(seg_lens) / fps), 3) if len(seg_lens) > 1 else 0.0,
        beat_sync, round(float(music.get("bpm") or 0) / 200.0, 4),
    ]

    # ---- 出镜人数统计 ----
    counts = [len(rec.get("people") or []) for rec in records]
    nonzero = [c for c in counts if c > 0]
    people_stats = {
        "min": int(min(nonzero)) if nonzero else 0,
        "max": int(max(counts)) if counts else 0,
        "median": int(np.median(nonzero)) if nonzero else 0,
        "mean": round(float(np.mean(counts)), 2) if counts else 0.0,
    }

    # ---- 朝向：模板限定横向/纵向（套用时同朝向才可用）----
    orientation = "landscape" if fw > fh else ("portrait" if fh > fw else "square")
    aspect_ratio = round(fw / max(1, fh), 4)

    # ---- 组装模板：默认打底 + 学到的字段覆盖 ----
    t = base
    t["meta"] = {"name": name, "genre": genre,
                 "n_regime": "group" if _looks_group(records) else "single",
                 "orientation": orientation,        # ★横向/纵向限定
                 "aspect_ratio": aspect_ratio,
                 "bpm": music.get("bpm"),
                 "bpm_tolerance": 25.0,
                 "people_count": people_stats,      # 出镜人数范围
                 # ★C位出镜情况（露出/遮挡）：套用时以此为主判别、优先保证
                 "c_exposure": c_exposure,          # full_body/half_body/head 占比
                 "shot_hist": shot_hist,            # 景别分布（按 C位露出统计）
                 "source": f"learned:{os.path.basename(ref_video)}"}
    print(f"[learn] 朝向={orientation}({fw}x{fh}) · 人数区间={people_stats}")
    print(f"[learn] C位出镜={c_exposure} · 景别直方图={shot_hist} · BPM={music.get('bpm')}")
    t["style"]["cut_rhythm_sec"] = cut_rhythm_sec
    t["style"]["shot_bias"] = shot_bias
    t["style"]["quantize_to_beat"] = bool(beat_sync >= 0.4)  # 参考本身卡点才开
    # ★参考本身是靠硬切换镜的 → 套用时也用硬切（裁剪窗瞬间跳），
    #   而不是把景别变化抹成缓慢推移。参考是长镜头则保持平滑过渡。
    t["style"]["hard_cut"] = bool(len(cuts) >= 3)
    t["section_default"] = section_default
    t["section_shot_dist"] = section_shot_dist     # ★景别分布，生成端据此派景别
    t["style_descriptor"] = style_descriptor

    # ★GROUP 景别下限 = 参考在「群体场景」下实际用过的最近景别。
    #   之前用 "近景占比>0.25 就放开" 的二元阈值太糙：参考真实用了 11% 中景，
    #   却被一刀切成 wide，那 11% 在套用时永远出不来。
    #   现在按数据说话：参考在 GROUP 帧上用过、且占比 >5% 的最近景别就是下限。
    g_tot = sum(group_shot.values())
    if g_tot > 0:
        floor = "wide"
        for s in tmpl.SHOT_LADDER:              # 近 → 远
            if group_shot.get(s, 0) / g_tot > 0.05:
                floor = s
                break
        t["style"]["group_min_shot"] = None if floor == "closeup" else floor
        g_dist = {k: round(v / g_tot, 3) for k, v in group_shot.most_common()}
        print(f"[learn] 群体帧景别={g_dist} → group_min_shot={t['style']['group_min_shot']}")
    else:
        t["style"]["group_min_shot"] = "wide"
    print(f"[learn] 段落景别分布={section_shot_dist}")
    # event_map / shot_coverage / accent / 仲裁权重：单参考学不可靠，保留默认（已编码标准）

    print(f"[learn] 换镜={len(cuts)} · cut_rhythm={cut_rhythm_sec}s · 卡点命中={beat_sync}")
    return t, records, music


def _looks_group(records):
    """粗判参考是否群舞：多数帧人数≥2 → group。"""
    multi = sum(1 for r in records if len(r.get("people") or []) >= 2)
    return multi > 0.5 * max(1, len(records))


def main():
    ap = argparse.ArgumentParser(description="运镜模板学习（离线，产出 template.json）")
    ap.add_argument("--ref-video", required=True, help="参考 MV 视频（用于换镜检测）")
    ap.add_argument("--analysis-dir", required=True,
                    help="该参考视频的分析产物目录（含 music.json / tracked_keypoints.json）")
    ap.add_argument("--out", required=True, help="输出 template.json 路径")
    ap.add_argument("--name", default="learned_v1")
    ap.add_argument("--genre", default="dance")
    ap.add_argument("--no-preview", action="store_true", help="不生成模板预览视频")
    ap.add_argument("--preview-seconds", type=float, default=None,
                    help="预览时长上限（秒）；缺省=与输入视频等长")
    args = ap.parse_args()

    t, records, music = learn(args.ref_video, args.analysis_dir, args.name, args.genre)
    tmpl.save_template(t, args.out)
    print(f"[learn] 模板已写出 -> {args.out}")

    # 学完顺手渲预览：用参考视频**自己的真实骨架+音乐**套模板，长度=输入视频长度
    if not args.no_preview:
        from . import preview
        preview_path = os.path.splitext(args.out)[0] + "_preview.mp4"
        real = preview.render_preview_from_data(
            args.out, preview_path, records, music,
            orientation=t["meta"].get("orientation", "portrait"),
            fps=float(music.get("fps") or 30.0),
            max_seconds=args.preview_seconds)
        if real:
            print(f"[learn] 模板预览 -> {real}")
        else:
            print("[learn][告警] 预览未生成（见上方 [预览] 告警）；模板本身已正常写出。")


if __name__ == "__main__":
    main()
