"""
模板 IR · 契约层（方案 §1.4 / §3 / §5 / §10）
=============================================
模板文件是「学习端」与「生成端」之间唯一的契约：
  - 学习端 learn_template.py：看参考视频 → 产出 template.json
  - 生成端 shotplan / camera ：读 template.json → 出分镜与相机

设计铁律：模板只存「与视频无关的相对量」——
  角色（C位）/ 相对时间（秒）/ 相对空间（画面比例）/ canonical 上的景别档位。
  绝对像素、帧号只在「套用那一刻」按当前视频翻译（frames=sec*fps，px=frac*dim）。

default_template() 把你的基本标准直接编码：
  · 节奏强→合重拍（accent.quantize_to_beat / downbeat 优先级）
  · 动作大→跟随（event_map: jump/big_move→extreme_wide+pull_out）
  · 舒缓→人占比多、活跃→稍多（section_default + 安静歌全局收敛）
  · 姿态→景别（全身动作→大远景/远景；上身→中景/特写）
  · 旋转优先级高（event_map: spin→roll, priority 高）★修旧代码漏接 spin
  · 景别留白（shot_coverage: 头顶1/4、脚下1/6、中/特写手臂出镜）
它同时是「兼容门控失败时的兜底通用模板」。
"""

import json
import os

# 统一景别枚举（近→远）。全系统以此为准，废弃旧的 upper 档。
SHOT_LADDER = ["closeup", "medium", "wide", "extreme_wide"]


# ===================================================================
# 默认模板：把基本标准编码进去（也是兼容门控的兜底）
# ===================================================================
def default_template():
    return {
        "meta": {"name": "generic_v1", "genre": "dance",
                 "n_regime": "single", "orientation": None,
                 "aspect_ratio": None, "source": "default"},

        # ---- 全局基调（无量纲，套用时翻译）----
        "style": {
            "cut_rhythm_sec": 1.0,          # 最短镜头时长（防碎切）
            "shot_bias": "medium",          # 缺省景别倾向
            "rotation_usage": 0.3,          # 旋转使用度 0..1（轻拍是否用 roll 等）
            "quantize_to_beat": True,       # 换镜吸附拍点
            # 换镜是否用硬切（裁剪窗瞬间跳到新构图）。false = 平滑过渡（长镜头风格）。
            "hard_cut": True,
            "prefer_downbeat_for_major": True,  # 重大切换优先落强拍
            "quiet_loudness_th": 0.08,      # 绝对响度低于此 → 判「安静歌」，全局收敛
            # 群体(GROUP)帧允许的最近景别：群舞时别切太近，保证队形。
            # 设为 null 可关闭该限制（允许 GROUP 段也给中景/特写）。
            "group_min_shot": "wide",
        },

        # ---- 段落基调：舒缓→人占比多(偏 follow/wide)，活跃→稍多运镜 ----
        # 注意：section 标签由 music 的绝对响度 + 相对能量共同判，避免慢歌也出 high
        "section_default": {
            "low":  {"shot": "wide",   "move": "follow"},
            "mid":  {"shot": "medium", "move": "follow"},
            "high": {"shot": "medium", "move": "push_in"},
        },

        # ---- 段落景别分布（学出的；空 = 退化成 section_default 的单一景别）----
        # 形如 {"high": {"wide": 0.5, "medium": 0.3, "closeup": 0.2}, ...}
        # 生成端按学到的换镜节奏切网格，再按此分布派景别 —— 这样才还原得出
        # "MV 在一个段落内在特写/中景/远景之间切换"的味道。
        # ★内置默认也要给景别分布，否则 shotplan 无配额可分 → 全片一个景别。
        #   （曾经 config.template.path 一空，成片就是 wide=100%，很难看出原因。）
        "section_shot_dist": {
            "low":  {"wide": 0.45, "extreme_wide": 0.15, "medium": 0.30, "closeup": 0.10},
            "mid":  {"wide": 0.45, "extreme_wide": 0.20, "medium": 0.25, "closeup": 0.10},
            "high": {"wide": 0.40, "extreme_wide": 0.30, "medium": 0.20, "closeup": 0.10},
        },

        # ---- 姿态事件 → 景别/运镜（含优先级与时长，秒）----
        # 全身动作(大跳/大位移)→大远景/远景；上身表达→中景/特写；旋转优先级高。
        # ---- 姿态事件 → 景别/运镜（含优先级与时长，秒）----
        # ★owns_shot：该事件是否有权夺取「景别」。
        #   true  = 这个动作在几何上必须要更松的景别才装得下（大跳/大位移/地面动作），
        #           不给就会出画 → 有权改景别。
        #   false = 表现性动作（旋转/伸展/手势/焦点转移），景别该由模板的段落分布决定，
        #           事件只贡献「运镜」。否则事件会铺满整条时间轴，
        #           把模板学到的景别分布整个盖掉，全片塌成一个景别。
        "event_map": {
            "jump":        {"shot": "extreme_wide", "move": "pull_out", "priority": 9, "dur_sec": 1.0, "owns_shot": True},
            "leap":        {"shot": "extreme_wide", "move": "pull_out", "priority": 9, "dur_sec": 1.0, "owns_shot": True},
            "big_move":    {"shot": "extreme_wide", "move": "pull_out", "priority": 8, "dur_sec": 1.0, "owns_shot": True},
            "travel":      {"shot": "extreme_wide", "move": "follow",   "priority": 8, "dur_sec": 1.0, "owns_shot": True},
            "level_change":{"shot": "wide",         "move": "pull_out", "priority": 8, "dur_sec": 1.1, "owns_shot": True},
            "floor":       {"shot": "wide",         "move": "follow",   "priority": 9, "dur_sec": 1.3, "owns_shot": True},
            "freeze":      {"shot": "wide",         "move": "static",   "priority": 8, "dur_sec": 0.8, "owns_shot": False},
            # 表现性动作：只出运镜，景别交给模板分布
            "spin":        {"shot": "medium",       "move": "roll",     "priority": 7, "dur_sec": 0.9, "owns_shot": False},
            "extension":   {"shot": "medium",       "move": "push_in",  "priority": 6, "dur_sec": 0.7, "owns_shot": False},
            "gesture":     {"shot": "medium",       "move": "recenter", "priority": 5, "dur_sec": 0.8, "owns_shot": False},
            "arm_hit":     {"shot": "medium",       "move": "push_in",  "priority": 6, "dur_sec": 0.7, "owns_shot": False},
            "face":        {"shot": "closeup",      "move": "recenter", "priority": 5, "dur_sec": 0.7, "owns_shot": True},
            "focus_switch":{"shot": "medium",       "move": "recenter", "priority": 4, "dur_sec": 0.6, "owns_shot": False},
        },

        # ---- 景别留白（你的规格，全部相对量）----
        # cover        : 目标覆盖率 = 内容高/裁剪窗高（camera 已用此定义）
        # span         : 取哪段身体（full 全身 / upper 髋以上 / chest 胸以上）
        # head_top_frac: 头顶到画面顶的留白占窗高比例（你的“头顶1/4/1/3”）
        # foot_bot_frac: 脚下到画面底的留白（仅 full 用，你的“脚下1/6”）
        "shot_coverage": {
            "extreme_wide": {"cover": 0.45, "span": "full",  "head_top_frac": 0.30, "foot_bot_frac": 0.20},
            "wide":         {"cover": 0.60, "span": "full",  "head_top_frac": 0.25, "foot_bot_frac": 0.1667},
            "medium":       {"cover": 0.82, "span": "upper", "head_top_frac": 0.28, "arms_in": True},
            "closeup":      {"cover": 1.05, "span": "chest", "head_top_frac": 0.25, "arms_in": True},
        },

        # ---- 强调层（重拍优先级高；轻拍按风格可有可无）----
        "accent": {
            "beat_pulse":     {"enabled": True, "delta_zoom": 0.045, "pre_f": 2, "post_f": 4},
            "downbeat_punch": {"enabled": True, "delta_zoom": 0.14,  "attack_f": 1, "release_f": 10},
            "sparsity_per_bar": 1.5,        # 每小节最多强调几下（治「太多」）
            # 局部对比 → 强调幅度 的响应形状（存形状不存绝对值，套用时重标定）
            "beat_response_curve": [[0.0, 0.0], [0.5, 0.3], [1.0, 1.0]],
            "target_peak_excursion": 0.18,  # 本片最强一下的目标视觉幅度
        },

        # ---- C位选择偏好（不同风格「看谁」不同）----
        "subject_weights": {"center": 0.9, "scale": 1.0, "motion": 0.8,
                            "frontal": 0.5, "event": 0.0},

        # ---- 多因素仲裁（§5）----
        "factor_weights": {"beat": 0.7, "section": 0.9, "pose_event": 1.0,
                          "focus_switch": 0.8, "silence": 0.5},
        "factor_envelopes": {
            "beat":         {"shape": "impulse", "pre": 0.07, "post": 0.15},
            "section":      {"shape": "step"},
            "pose_event":   {"shape": "bell", "pre": 0.25, "post": 0.8},
            "focus_switch": {"shape": "ramp", "post": 0.5},
        },

        # ---- 风格指纹（可检索；单参考主要靠它 + accent 承载风格）----
        "style_descriptor": [],

        # ---- 安全/构图硬约束的相对量 ----
        "constraints": {"safe_frame_margin": 0.06, "core_y_anchor": 0.42},
    }


# ===================================================================
# 读 / 写
# ===================================================================
def load_template(path):
    """读模板文件；缺省或读失败 → 返回默认模板（永不让生成端拿到空）。"""
    if not path or not os.path.exists(path):
        return default_template()
    try:
        with open(path, encoding="utf-8") as f:
            t = json.load(f)
    except Exception as e:
        print(f"[template][告警] 读 {path} 失败({e})，回退默认模板")
        return default_template()
    return _fill_missing(t, default_template())


def save_template(t, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(t, f, ensure_ascii=False, indent=2)
    return path


def _fill_missing(t, base):
    """学习端可能只产出部分字段；缺的用默认补齐（深合并），保证生成端字段完整。"""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (t or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _fill_missing(v, out[k])
        else:
            out[k] = v
    return out


# ===================================================================
# 套用期：把模板的相对量注入生成端配置（camera 几乎不用改，读 config 即可）
# ===================================================================
def merge_into_config(config, template):
    """
    把模板里 camera 需要的相对量注入 config['camera']，让 camera 原样读取：
      - shot_coverage（景别留白）
      - safe_frame.core_y_anchor / margin_frac
      - templates.beat_pulse / downbeat_punch（强调层参数）
    shotplan 则直接读 ctx.extras['template']（section_default/event_map/style）。
    """
    cfg = dict(config)
    cam = dict(cfg.get("camera", {}))
    # 原版 camera._shot_targets 读 float(cover)，所以给它拍平后的覆盖率；
    # 完整留白（head_top_frac/span/foot_bot_frac）另存 shot_composition，
    # 供打了「构图补丁」的 camera 读取（缺则原版行为，不崩）。
    raw_cov = template.get("shot_coverage", {}) or {}
    cam["shot_coverage"] = {k: (v.get("cover") if isinstance(v, dict) else float(v))
                            for k, v in raw_cov.items()}
    cam["shot_composition"] = raw_cov

    sf = dict(cam.get("safe_frame", {}))
    cons = template.get("constraints", {})
    sf.setdefault("core_y_anchor", cons.get("core_y_anchor", 0.42))
    sf.setdefault("margin_frac", cons.get("safe_frame_margin", 0.06))
    cam["safe_frame"] = sf

    tpls = dict(cam.get("templates", {}))
    acc = template.get("accent", {})
    tpls["beat_pulse"] = acc.get("beat_pulse", tpls.get("beat_pulse", {}))
    tpls["downbeat_punch"] = acc.get("downbeat_punch", tpls.get("downbeat_punch", {}))
    cam["templates"] = tpls

    cfg["camera"] = cam
    return cfg


# ===================================================================
# 兼容门控 + 重标定（§10）——占位骨架，接入学习式模板后补全
# ===================================================================
def video_stats_from(records, music):
    """从目标视频的 records + music 抽出兼容判别所需统计。"""
    import numpy as np
    counts = [len(r.get("people") or []) for r in (records or [])]
    nz = [c for c in counts if c > 0]
    median = int(np.median(nz)) if nz else 0
    orientation = None
    if records:
        sh = records[0].get("original_shape") or []
        if len(sh) >= 2:
            h, w = sh[0], sh[1]
            orientation = "landscape" if w > h else ("portrait" if h > w else "square")
    return {
        "n_regime": "group" if (nz and (sum(1 for c in counts if c >= 2) > 0.5 * len(counts))) else "single",
        "people_median": median,
        "people_max": int(max(counts)) if counts else 0,
        "orientation": orientation,
        "bpm": (music or {}).get("bpm"),
    }


def compatibility(template, video_stats):
    """
    compat(T,V) ∈ [0,1]。先硬否决不可行组合，再对可行的算软距离。
    硬否决（返回 0）：
      · 群舞模板套独舞（target=group 的动作在独舞上物理不存在）
      · 目标人数落在模板人数范围之外太远
    软距离：BPM 偏差、人数中位数偏差、（可选）响度。
    """
    import math
    meta = template.get("meta", {})

    # --- 硬否决 ---
    if meta.get("n_regime") == "group" and video_stats.get("n_regime") == "single":
        return 0.0
    # 朝向不一致（横向模板套竖屏，反之亦然）→ 构图/留白根本不同，硬否决
    o = meta.get("orientation")
    if o and video_stats.get("orientation") and o != video_stats["orientation"]:
        return 0.0
    pc = meta.get("people_count") or {}
    vmed = video_stats.get("people_median")
    if pc and vmed is not None and pc.get("max"):
        # 目标人数超过模板见过的最多人数 +2，或不足最少人数 -1，判为不可行
        if vmed > pc.get("max", 99) + 2 or vmed < max(1, pc.get("min", 1)) - 1:
            return 0.0

    # --- 软距离 ---
    D = 0.0
    ref_bpm = meta.get("bpm")
    tol = float(meta.get("bpm_tolerance", 25.0))
    if ref_bpm and video_stats.get("bpm"):
        # 超出容差的部分才计距离（容差内视为完全匹配）
        d_bpm = max(0.0, abs(video_stats["bpm"] - ref_bpm) - tol) / max(1.0, ref_bpm)
        D += 1.0 * d_bpm
    if pc.get("median") and vmed is not None:
        D += 0.5 * abs(vmed - pc["median"]) / max(1.0, pc["median"])
    return math.exp(-D)
