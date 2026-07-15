"""
阶段 3：导演层 (ShotPlan) · 模板驱动重写
========================================
读取模板文件（template.json），据此出分镜表。旧的硬编码 DEFAULT_SECTION_DEFAULT /
DEFAULT_EVENT_MAP 已下沉到 template.default_template()，本层不再自带策略常量——
「策略」全部来自模板，「机制」（铺基线→叠事件→防碎→吸拍）留在这里。

数据流：
  ctx.extras['music']  段落/拍点/能量（+ abs_loudness 判安静歌）
  ctx.extras['pose']   姿态事件（含 subject 层注入的 focus_switch）
  template             section_default / event_map / style / accent
        │
        ├─ 铺段落基线（低优先）→ 叠事件（高优先，含 spin/extension/focus_switch）
        ├─ 安静歌全局收敛（舒缓→人占比多）
        ├─ GROUP 帧偏 wide（多人队形）
        ├─ 合并成段 → 防碎（cut_rhythm_sec）→ 换镜吸附拍点（重大切换优先强拍）
        └─ 写 shot_plan.json；并把模板相对量注入 config 供 camera 读取

同时负责：template.load + merge_into_config（camera 读 shot_coverage/safe_frame/accent）。
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np

from ..stage import Stage, register
from .. import template as tmpl

SHOT_LADDER = tmpl.SHOT_LADDER   # 近 → 远


# 旧别名兼容（万一外部配置仍用旧名）
SHOT_ALIASES = {"upper": "medium", "long": "wide", "full": "wide",
                "extreme": "extreme_wide", "extreme-wide": "extreme_wide",
                "big_wide": "extreme_wide"}


def _norm_shot(s):
    s = str(s or "medium")
    return SHOT_ALIASES.get(s, s)


def _section_label_at(sections, f):
    for s in sections:
        if s["start_f"] <= f < s["end_f"]:
            return s.get("label", "mid")
    return "mid"


def _event_dur_frames(spec, fps, intensity):
    d = float(spec.get("dur_sec", 0.9)) * (0.85 + 0.4 * min(1.0, float(intensity)))
    return max(1, int(round(d * fps)))


def _nearest_beat(f, beat_frames, window, prefer=None):
    if prefer is not None and len(prefer):
        cand = prefer[np.abs(prefer - f) <= window]
        if len(cand):
            return int(cand[np.argmin(np.abs(cand - f))])
    if beat_frames is not None and len(beat_frames):
        cand = beat_frames[np.abs(beat_frames - f) <= window]
        if len(cand):
            return int(cand[np.argmin(np.abs(cand - f))])
    return int(f)


def _build_cut_grid(n, fps, cut_rhythm_sec, beat_frames, down_frames, prefer_down):
    """
    按学到的换镜节奏切一条镜头网格，边界吸附拍点（重大切换优先强拍）。
    这是"模板驱动分镜"的骨架：先决定何时换镜，再决定每段给什么景别。
    """
    step = max(2, int(round(float(cut_rhythm_sec) * fps)))
    win = max(1, int(round(0.2 * fps)))
    bounds = [0]
    f = step
    while f < n - step // 2:
        snap = _nearest_beat(f, beat_frames, win,
                             prefer=down_frames if prefer_down else None)
        snap = max(bounds[-1] + max(3, step // 3), min(int(snap), n - 1))
        if snap > bounds[-1]:
            bounds.append(snap)
        f = max(snap + step, f + step)
    bounds.append(n)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]


# 事件设计景别 → 近景亲和度。复用 event_map 里已有的 shot 字段，不另立参数。
_AFFINITY = {"closeup": 2.0, "medium": 1.0, "wide": -1.0, "extreme_wide": -2.0}


def _segment_content(grid, events, event_map, fps, n):
    """
    ★本片事件 = 模板套用的「参考」与「兜底」，不是与模板竞争的对手。

    模板回答「多少」（68% 远景、8% 中景），本片事件回答「哪一段」：
      · 近景亲和度 affinity：这一段里在发生什么，适合更近还是更松。
        上身表现(extension/gesture/face) → 适合近；全身大动作 → 适合松。
      · 几何兜底 forced：大跳/大位移/地面动作**必须**更松才装得下，
        否则会出画 —— 这一段无论模板配额如何都强制给该事件的景别。

    返回 (affinity[], forced_shot[])，长度与 grid 相同。
    """
    aff = [0.0] * len(grid)
    forced = [None] * len(grid)
    for ev in events:
        f = int(ev.get("frame", -1))
        if not (0 <= f < n):
            continue
        spec = event_map.get(str(ev.get("type", "")))
        if not spec or not bool(spec.get("enabled", True)):
            continue
        prio = int(spec.get("priority", 5))
        dur = _event_dur_frames(spec, fps, ev.get("intensity", 0.5))
        s, e = max(0, f - int(dur * 0.25)), min(n, f + dur)
        ev_shot = _norm_shot(spec.get("shot", "medium"))
        w = _AFFINITY.get(ev_shot, 0.0) * (0.5 + float(ev.get("intensity", 0.5)))
        owns = bool(spec.get("owns_shot", prio >= 8))
        for gi, (gs, ge) in enumerate(grid):
            ov = min(e, ge) - max(s, gs)
            if ov <= 0:
                continue
            frac = ov / max(1, ge - gs)          # 事件覆盖该段的比例
            aff[gi] += w * frac
            if owns and frac > 0.15:
                # 几何必需：取更松的那个（多个强制事件时以最松为准）
                cur = forced[gi]
                if cur is None or SHOT_LADDER.index(ev_shot) > SHOT_LADDER.index(cur):
                    forced[gi] = ev_shot
    return aff, forced


def _quota_counts(dist, m):
    """把景别分布换算成 m 段的整数配额（最大余数法，保证总数正好 m）。"""
    raw = {s: float(p) * m for s, p in dist.items()}
    cnt = {s: int(np.floor(v)) for s, v in raw.items()}
    rest = m - sum(cnt.values())
    for s, _ in sorted(raw.items(), key=lambda kv: kv[1] - np.floor(kv[1]), reverse=True):
        if rest <= 0:
            break
        cnt[s] += 1
        rest -= 1
    return cnt


def _assign_shots(grid, sections, dist_by_lab, default_by_lab, aff, forced):
    """
    ★模板定「多少」，本片事件定「哪一段」。

    1) forced 段（大跳/地面动作等几何必需）先落位——安全兜底，不参与配额争夺；
    2) 其余段按「近景亲和度」从高到低排序，把模板配额**从近到远**依次发下去。
       → 特写发给最适合特写的那几段（上身表现最强），远景发给最松的那几段。
    这样模板的比例被还原，而"哪一段给什么"由本片内容说了算。
    """
    out = [None] * len(grid)
    by_lab = defaultdict(list)
    for gi, (s, e) in enumerate(grid):
        by_lab[_section_label_at(sections, (s + e) // 2)].append(gi)

    for lab, idxs in by_lab.items():
        dist = dist_by_lab.get(lab) or {}
        if not dist:
            sh = _norm_shot(default_by_lab.get(lab, {}).get("shot", "medium"))
            for gi in idxs:
                out[gi] = forced[gi] or sh
            continue

        free = [gi for gi in idxs if forced[gi] is None]
        # ★配额按「全部段」算，几何强制段先落位并**从配额里扣除**，
        #   再把剩余配额归一到自由段——否则强制段等于外挂，全片比例会被挤偏。
        cnt = _quota_counts(dist, len(idxs))
        for gi in idxs:
            if forced[gi] is not None:
                out[gi] = forced[gi]
                cnt[forced[gi]] = max(0, cnt.get(forced[gi], 0) - 1)
        total = sum(cnt.values())
        while total > len(free):          # 配额多了：从占比最小的景别扣
            for sh in sorted(dist, key=lambda x: dist[x]):
                if cnt.get(sh, 0) > 0:
                    cnt[sh] -= 1; total -= 1; break
            else:
                break
        while total < len(free):          # 配额少了：补给占比最大的景别
            sh = max(dist, key=lambda x: dist[x])
            cnt[sh] = cnt.get(sh, 0) + 1; total += 1

        # 亲和度高的段优先拿更近的景别
        free.sort(key=lambda gi: aff[gi], reverse=True)
        pos = 0
        for sh in SHOT_LADDER:                # 近 → 远
            for _ in range(cnt.get(sh, 0)):
                if pos >= len(free):
                    break
                out[free[pos]] = sh
                pos += 1
        for gi in free[pos:]:                 # 兜底
            out[gi] = _norm_shot(default_by_lab.get(lab, {}).get("shot", "medium"))
    return [(grid[i][0], grid[i][1], out[i] or "medium") for i in range(len(grid))]


@register("shotplan")
class ShotPlanStage(Stage):
    name = "shotplan"

    def run(self, ctx):
        cfg = ctx.config.get("shotplan", {})
        n = int(ctx.timeline.frame_count)
        fps = float(ctx.timeline.fps)

        # ---- 载入模板（生成端读取模板文件的入口）----
        tpath = ctx.config.get("template", {}).get("path")
        template = tmpl.load_template(tpath)
        ctx.extras["template"] = template
        # 把模板相对量注入 config，camera 后续原样读取（shot_coverage/safe_frame/accent）
        ctx.config = tmpl.merge_into_config(ctx.config, template)
        print(f"[shotplan] 模板={template['meta'].get('name')} "
              f"(source={template['meta'].get('source')})")

        if not cfg.get("enabled", True):
            ctx.extras["shot_plan"] = [{"start_f": 0, "end_f": n, "shot": "medium",
                                        "move": "follow", "priority": 0,
                                        "src": "disabled", "intensity": 0.0}]
            print("[shotplan] 已关闭：全片 medium/follow")
            return ctx

        music = ctx.extras.get("music", {}) or {}
        pose = ctx.extras.get("pose", {}) or {}
        prim = ctx.extras.get("primary_records") or []
        sections = music.get("sections") or [{"start_f": 0, "end_f": n, "label": "mid"}]
        beat_grid = music.get("beat_grid") or []
        events = pose.get("pose_events") or []

        # 兼容判别：拿目标视频统计算 compat，低分给出建议（软提示，不硬拦）
        v_stats = tmpl.video_stats_from(prim, music)
        compat = tmpl.compatibility(template, v_stats)
        if template["meta"].get("source", "default") != "default":
            tag = "可用" if compat >= 0.6 else ("勉强" if compat >= 0.3 else "不建议")
            print(f"[shotplan] 兼容分={compat:.2f}（{tag}）· 目标 regime={v_stats['n_regime']} "
                  f"朝向={v_stats.get('orientation')} 人数中位={v_stats['people_median']} BPM={v_stats.get('bpm')} "
                  f"| 模板 regime={template['meta'].get('n_regime')} "
                  f"朝向={template['meta'].get('orientation')} "
                  f"人数={template['meta'].get('people_count')} BPM={template['meta'].get('bpm')}")
            if compat < 0.3:
                print("[shotplan][告警] 模板与该视频差异较大（人数/节奏），效果可能不佳；"
                      "可换更合适模板或用内置默认模板。")

        beat_frames = np.array([int(b["frame"]) for b in beat_grid
                                if 0 <= int(b["frame"]) < n], dtype=int)
        down_frames = np.array([int(b["frame"]) for b in beat_grid
                                if b.get("is_downbeat") and 0 <= int(b["frame"]) < n], dtype=int)

        style = template["style"]
        section_default = template["section_default"]
        event_map = template["event_map"]
        min_shot = max(1, int(round(float(style.get("cut_rhythm_sec", 1.0)) * fps)))
        quantize = bool(style.get("quantize_to_beat", True))
        prefer_down = bool(style.get("prefer_downbeat_for_major", True))
        beat_win = max(1, int(round(0.15 * fps)))

        # ---- 安静歌判定（舒缓→人占比多、少运镜）----
        # music 需产出 abs_loudness（见 music.py 补丁）；缺省则不触发收敛。
        quiet = float(music.get("abs_loudness", 1.0)) < float(style.get("quiet_loudness_th", 0.0))
        if quiet:
            print("[shotplan] 判为安静歌：全局偏 wide/follow、抑制 push_in")

        # ---- GROUP 帧掩码（多人队形 → 偏 wide）----
        group_mask = np.zeros(n, dtype=bool)
        for r in prim:
            f = r.get("frame_index")
            if f is None or not (0 <= int(f) < n):
                continue
            if r.get("compose_mode") == "GROUP":
                group_mask[int(f)] = True

        # GROUP 帧允许的最近景别（模板旋钮；null=不限制）
        group_floor = style.get("group_min_shot", "wide")
        _ladder = tmpl.SHOT_LADDER   # 近→远
        def _apply_group_floor(shot):
            """把景别拉到不比 group_floor 更近（群舞保队形）。"""
            if not group_floor or shot not in _ladder or group_floor not in _ladder:
                return shot
            return group_floor if _ladder.index(shot) < _ladder.index(group_floor) else shot

        # ---- 铺基线（低优先级）----
        # 有学到的景别分布 → 按学到的换镜节奏切网格 + 按分布派景别（还原 MV 的景别变化）
        # 没有            → 退化成每帧读 section_default 的单一景别（旧行为）
        frame_shot = np.empty(n, dtype=object)
        dist_by_lab = template.get("section_shot_dist") or {}
        if dist_by_lab:
            grid = _build_cut_grid(n, fps, style.get("cut_rhythm_sec", 1.0),
                                   beat_frames, down_frames, prefer_down)
            # ★本片事件作为参考与兜底：算每段的近景亲和度 + 几何必需的强制景别
            aff, forced = _segment_content(grid, events, event_map, fps, n)
            assigned = _assign_shots(grid, sections, dist_by_lab, section_default, aff, forced)
            nf = sum(1 for x in forced if x)
            print(f"[shotplan] 模板驱动分镜：{len(grid)} 个镜头网格 @ {style.get('cut_rhythm_sec')}s "
                  f"· 模板定配额={dist_by_lab} · 本片事件定哪一段（{nf} 段被大动作强制给松景别）")
            for (s, e, shot) in assigned:
                lab = _section_label_at(sections, (s + e) // 2)
                move = section_default.get(lab, {}).get("move", "follow")
                if quiet and move == "push_in":
                    move = "follow"
                sh = shot
                if quiet and sh in ("closeup", "medium"):
                    sh = "wide"                       # 安静歌：人占比多、别乱推近
                for i in range(s, min(e, n)):
                    frame_shot[i] = (0, (_apply_group_floor(sh) if group_mask[i] else sh),
                                     move, f"tpl:{lab}", 0.0)
        else:
            for i in range(n):
                lab = _section_label_at(sections, i)
                d = section_default.get(lab, {"shot": "medium", "move": "follow"})
                shot, move = _norm_shot(d.get("shot", "medium")), d.get("move", "follow")
                if quiet and move == "push_in":       # 安静歌：不主动推近
                    move, shot = "follow", ("wide" if shot == "medium" else shot)
                if group_mask[i]:
                    shot = _apply_group_floor(shot)
                frame_shot[i] = (0, shot, move, f"section:{lab}", 0.0)
        for i in range(n):                            # 兜底
            if frame_shot[i] is None:
                frame_shot[i] = (0, "medium", "follow", "fallback", 0.0)

        # ---- 叠事件（高优先级；含 spin/extension/focus_switch）----
        # ★owns_shot=False 的事件只贡献「运镜」，不夺取「景别」——景别归模板的段落分布。
        #   否则表现性事件（旋转/伸展/焦点转移）会铺满整条时间轴，
        #   把模板学到的景别分布整个盖掉（表现为"模板没效果、全片一个景别"）。
        ev_cover = np.zeros(n, dtype=bool)
        for ev in events:
            f = int(ev.get("frame", -1))
            if not (0 <= f < n):
                continue
            spec = event_map.get(str(ev.get("type", "")))
            if not spec or not bool(spec.get("enabled", True)):
                continue
            prio = int(spec.get("priority", 5))
            owns_shot = bool(spec.get("owns_shot", prio >= 8))
            dur = _event_dur_frames(spec, fps, ev.get("intensity", 0.5))
            s = max(0, f - int(dur * 0.25))
            e = min(n, f + dur)
            ev_cover[s:e] = True
            ev_shot = _norm_shot(spec.get("shot", "medium"))
            move = spec.get("move", "follow")
            src = f"event:{ev.get('type')}"
            inten = float(ev.get("intensity", 0.5))
            for i in range(s, e):
                if prio < frame_shot[i][0]:
                    continue
                if owns_shot and not dist_by_lab:
                    # 无模板分布（内置默认模板）：几何必需的事件仍可夺取景别
                    sh = _apply_group_floor(ev_shot) if group_mask[i] else ev_shot
                else:
                    # 有模板分布：景别已在配额阶段由「模板定多少 + 本片事件定哪一段」
                    # 共同决定（含几何强制），这里不再改景别，只贡献运镜。
                    sh = frame_shot[i][1]
                frame_shot[i] = (prio, sh, move, src, inten)
        print(f"[shotplan] 事件覆盖 {100.0*ev_cover.mean():.0f}% 帧"
              f"（其中仅改运镜、不改景别的事件保留模板景别）")

        # ---- 防碎切：★只对「景别」施加最短时长 ----
        # min_shot 是"换镜节奏"，管的是景别切换。若把它套在 (景别,运镜) 组合上，
        # 事件驱动的运镜变化（~25 帧）会被整段并掉 —— 表现为"只剩几段、没有运镜"。
        # 电影语言里：一个镜头内部本就可以推/摇/滚，运镜变化不是换镜。
        shot_runs = []
        i = 0
        while i < n:
            j = i
            while j + 1 < n and frame_shot[j + 1][1] == frame_shot[i][1]:
                j += 1
            prio = max(frame_shot[k][0] for k in range(i, j + 1))
            shot_runs.append([i, j + 1, (prio, frame_shot[i][1])])
            i = j + 1
        shot_runs = self._enforce_min_shot(shot_runs, min_shot)
        for (s, e, c) in shot_runs:            # 把并完的景别写回逐帧
            for i in range(int(s), int(e)):
                fs = frame_shot[i]
                frame_shot[i] = (fs[0], c[1], fs[2], fs[3], fs[4])

        # ---- 再按 (景别, 运镜) 切最终段；只压掉极短的运镜抖动 ----
        raw = []
        i = 0
        while i < n:
            cur = frame_shot[i]
            j = i
            while j + 1 < n and frame_shot[j + 1][1] == cur[1] and frame_shot[j + 1][2] == cur[2]:
                j += 1
            raw.append([i, j + 1, cur])
            i = j + 1
        segs = self._enforce_min_shot(raw, max(2, int(round(0.2 * fps))))

        # ---- 换镜边界吸附拍点（重大切换优先强拍）----
        if quantize and (len(beat_frames) or len(down_frames)):
            for k in range(1, len(segs)):
                b = segs[k][0]
                major = abs(segs[k][2][0] - segs[k - 1][2][0]) >= 2
                snap = _nearest_beat(b, beat_frames, beat_win,
                                     prefer=down_frames if (major and prefer_down) else None)
                snap = max(segs[k - 1][0] + 1, min(snap, segs[k][1] - 1))
                segs[k][0] = snap
                segs[k - 1][1] = snap

        plan = [{"start_f": int(s), "end_f": int(e), "shot": c[1], "move": c[2],
                 "priority": int(c[0]), "src": c[3], "intensity": round(float(c[4]), 3)}
                for s, e, c in segs]
        # ★标记硬切：景别发生变化处就是换镜。camera 会在这些帧让裁剪窗瞬间跳过去，
        #   否则 One Euro + 限速会把切镜抹成一段缓慢推移（表现为"全是移动运镜、没有切"）。
        hard_cut = bool(style.get("hard_cut", True))
        n_cut = 0
        for k, p_ in enumerate(plan):
            p_["cut"] = False
            if k == 0 or not hard_cut:
                continue
            if p_["shot"] != plan[k - 1]["shot"]:
                p_["cut"] = True; n_cut += 1
        print(f"[shotplan] 硬切 {n_cut} 处（hard_cut={hard_cut}）")
        ctx.extras["shot_plan"] = plan

        out_dir = ctx.extras.get("analysis_out_dir", ".")
        os.makedirs(out_dir, exist_ok=True)
        # ★同时报「逐帧占比」：段数会被运镜变化切碎（一个 wide 镜头内推/摇/滚会分成多段），
        #   只看段数会严重误判景别比例。真正该对比模板配额的是逐帧占比。
        _fr = np.empty(n, dtype=object)
        for p_ in plan:
            for _i in range(p_["start_f"], min(p_["end_f"], n)):
                _fr[_i] = p_["shot"]
        _c = Counter(x for x in _fr if x)
        _tot = sum(_c.values()) or 1
        _pct = " ".join(f"{k}={v/_tot:.3f}" for k, v in _c.most_common())
        print(f"[shotplan] 逐帧景别占比: {_pct}")
        if dist_by_lab:
            _tgt = " ".join(f"{k}={v:.3f}" for k, v in
                            sorted(next(iter(dist_by_lab.values())).items(),
                                   key=lambda kv: -kv[1]))
            print(f"[shotplan] 模板目标占比: {_tgt}")

        with open(os.path.join(out_dir, "shot_plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        sc, mc = Counter(p["shot"] for p in plan), Counter(p["move"] for p in plan)
        print(f"[shotplan] {len(plan)} 段 · 景别{dict(sc)} · 运动{dict(mc)} · "
              f"拍点{len(beat_frames)}(强拍{len(down_frames)}) "
              f"· GROUP帧{int(group_mask.sum())} -> shot_plan.json")
        return ctx

    @staticmethod
    def _enforce_min_shot(raw_segs, min_shot):
        segs = [list(s) for s in raw_segs]
        changed = True
        while changed and len(segs) > 1:
            changed = False
            for k in range(len(segs)):
                s, e, c = segs[k]
                if e - s >= min_shot:
                    continue
                left = segs[k - 1] if k > 0 else None
                right = segs[k + 1] if k + 1 < len(segs) else None
                if left and right:
                    if left[2][0] != right[2][0]:
                        target = left if left[2][0] > right[2][0] else right
                    else:
                        target = left if (left[1] - left[0]) >= (right[1] - right[0]) else right
                else:
                    target = left or right
                if target is left:
                    left[1] = e
                else:
                    right[0] = s
                segs.pop(k)
                changed = True
                break
        return segs
