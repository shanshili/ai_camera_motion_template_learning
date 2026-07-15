"""
阶段 4：相机 (Camera) · WP4-6 改造版 + 安全框硬约束
=====================================================
不再自己决定表达，而是把导演层的分镜表翻译成逐帧相机参数。
  - follow 基线：主体跟随 + One Euro + 死区 + 限速（稳定底座）
  - 景别 shot：wide/medium/upper/closeup → 目标覆盖率(zoom) + 构图锚点(center)
  - 运动 move：static/follow/push_in/pull_out/roll/orbit/recenter → zoom/center/rot 包络
  - 拍点修饰：beat_pulse/downbeat_punch 叠加，幅度按能量调制；freeze 段抑制大脉冲
  - ★安全框硬约束：无论运镜怎么算，最终裁剪窗必须包住"该露的关键点"
      · 默认：上半身（头顶+双肩+髋）必须出镜
      · 低重心事件（level_change/freeze/...）：全身（加双踝）必须出镜
      · 包不下则强制压低 zoom（拉远）；连全身都超出源画面则顶到 1.0 不再拉远
导出 camera_track.json / camera_keyframes.json / metrics.json。
"""

import json
import math
import os

import numpy as np

from ..stage import Stage, register
from ..context import CameraParams
from ..transform import (largest_source_view, crop_rect, effective_max_zoom,
                         clamp)
from ..pose_events import OneEuro
from ..log import log, Progress


def _lerp(a, b, t):
    return a + (b - a) * t


def _smoothstep(t):
    t = clamp(float(t), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _interp_missing(values):
    arr = np.asarray(values, dtype=float)
    good = np.isfinite(arr)
    if good.all():
        return arr
    if not good.any():
        return np.zeros(len(arr), dtype=float)
    idx = np.arange(len(arr))
    arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    return arr


def _safe(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _interp_missing_keep_allnan(values):
    """
    与 _interp_missing 类似，但如果整段序列全是 NaN（该视频从没检测到这个
    关键点集合），保留 NaN 而不是退化成全 0 —— 全 0 会让 _safe_box 误判成
    "检测到一个贴在原点的框"，比"没检测到、走 bbox 回退"更危险。
    """
    arr = np.asarray(values, dtype=float)
    good = np.isfinite(arr)
    if good.all():
        return arr
    if not good.any():
        return arr
    idx = np.arange(len(arr))
    arr = arr.copy()
    arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    return arr


# ===================================================================
# ★新增：安全框极值序列的零相位轻平滑（修开头/全程抖动的根因之一）
# ===================================================================
def _light_smooth(arr, fps, win_sec=0.12):
    """
    对逐帧关键点极值序列做零相位滑动平均，压掉"这一帧检测到、下一帧没检测到/
    置信度切换导致的抖动"，同时保留"从未检测到任何关键点"的位置为 NaN
    （交给 _safe_box 的 bbox 回退逻辑处理，不瞎补一个假值）。
    这是离线处理（geom 一次性构建好），可以用未来帧，所以用滑动平均而不是
    因果的 One Euro，不会引入额外相位延迟。
    """
    arr = np.asarray(arr, dtype=float)
    nan_mask = ~np.isfinite(arr)
    if nan_mask.all():
        return arr  # 全程无检测：原样返回 NaN
    win = max(1, int(round(win_sec * fps)))
    if win <= 1:
        return arr
    filled = arr.copy()
    if nan_mask.any():
        idx = np.arange(len(arr))
        good = ~nan_mask
        filled[nan_mask] = np.interp(idx[nan_mask], idx[good], arr[good])
    kernel = np.ones(win) / win
    pad = win // 2
    padded = np.pad(filled, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")[:len(arr)]
    smoothed[nan_mask] = np.nan  # 从未检测到的帧仍保留 NaN，走 fallback
    return smoothed


# ===================================================================
# ★新增：低重心事件 → 每帧是否要求"全身出镜"的掩码
# ===================================================================
def _fullbody_mask(ctx, n, fps, cfg):
    """
    根据 pose_events 里的低重心事件，标出哪些帧需要全身出镜。
    事件类型可在 camera.safe_frame.fullbody_events 配置；默认 level_change/freeze。
    每个事件以其 frame 为中心，向前后各扩 pad_sec 秒，覆盖整个地板动作过程。
    """
    sf_cfg = cfg.get("safe_frame", {})
    ev_types = set(sf_cfg.get("fullbody_events",
                              ["level_change", "freeze", "floor", "ground", "crouch"]))
    pad_sec = float(sf_cfg.get("fullbody_pad_sec", 0.6))
    pad = max(1, int(round(pad_sec * fps)))

    mask = np.zeros(n, dtype=bool)
    pose = ctx.extras.get("pose", {}) or {}
    for ev in pose.get("pose_events", []):
        if str(ev.get("type", "")) not in ev_types:
            continue
        f = int(ev.get("frame", -1))
        if not (0 <= f < n):
            continue
        s = max(0, f - pad)
        e = min(n, f + pad + 1)
        mask[s:e] = True
    return mask


# ----------------------------- 主体几何序列 -----------------------------
def _subject_geometry(ctx, n):
    """
    逐帧主体几何：bbox 中心/宽高 + 关节锚点（肩中点、鼻）。
    优先用 primary_records 的关键点，退回 pose.json 的 bbox。
    ★额外返回安全框所需的关键点极值：头顶、双肩、髋、双踝 的 x/y 范围。
    返回 dict of np.array，缺失已插值补齐（安全框极值保留 NaN，装配时按需回退 bbox）。
    """
    src_w = int(ctx.meta["width"]); src_h = int(ctx.meta["height"])
    pose = ctx.extras.get("pose", {}) or {}
    prim = ctx.extras.get("primary_records") or []
    prim_by_f = {int(r["frame_index"]): r.get("primary_person") for r in prim
                 if r.get("frame_index") is not None}

    bx = np.full(n, np.nan); by = np.full(n, np.nan)
    bw = np.full(n, np.nan); bh = np.full(n, np.nan)
    head_y = np.full(n, np.nan); sh_x = np.full(n, np.nan); sh_y = np.full(n, np.nan)
    hip_y = np.full(n, np.nan)

    # ★安全框极值序列（源坐标系，NaN 表示该帧无此关键点）
    hc_x0 = np.full(n, np.nan); hc_x1 = np.full(n, np.nan)   # ★头胸核心 x 范围（最高优先）
    hc_y0 = np.full(n, np.nan); hc_y1 = np.full(n, np.nan)   # ★头胸核心 y 范围（头顶→胸/肩下）
    up_x0 = np.full(n, np.nan); up_x1 = np.full(n, np.nan)   # 上半身 x 范围
    up_y0 = np.full(n, np.nan); up_y1 = np.full(n, np.nan)   # 上半身 y 范围（头顶→髋）
    fb_x0 = np.full(n, np.nan); fb_x1 = np.full(n, np.nan)   # 全身 x 范围
    fb_y0 = np.full(n, np.nan); fb_y1 = np.full(n, np.nan)   # 全身 y 范围（头顶→踝）

    for pf in pose.get("pose_frames", []):
        f = int(pf.get("frame", -1))
        if not (0 <= f < n):
            continue
        b = pf.get("bbox")
        if b:
            x, y, w, h = map(float, b)
            bx[f] = x + w / 2.0; by[f] = y + h / 2.0
            bw[f] = max(1.0, w); bh[f] = max(1.0, h)

    def _kp(person, name, conf_th=0.2):
        for kp in (person.get("keypoints") or []):
            if kp.get("name") == name and kp.get("xy"):
                c = kp.get("confidence")
                if c is None or c >= conf_th:
                    return float(kp["xy"][0]), float(kp["xy"][1])
        return None

    def _minmax(pts):
        pts = [p for p in pts if p is not None]
        if not pts:
            return None, None, None, None
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return min(xs), max(xs), min(ys), max(ys)

    for f in range(n):
        p = prim_by_f.get(f)
        if not p:
            continue
        ls = _kp(p, "left_shoulder"); rs = _kp(p, "right_shoulder")
        nose = _kp(p, "nose")
        leye = _kp(p, "left_eye"); reye = _kp(p, "right_eye")
        lear = _kp(p, "left_ear"); rear = _kp(p, "right_ear")
        lh = _kp(p, "left_hip"); rh = _kp(p, "right_hip")
        la = _kp(p, "left_ankle"); ra = _kp(p, "right_ankle")
        lk = _kp(p, "left_knee"); rk = _kp(p, "right_knee")
        lw = _kp(p, "left_wrist"); rw = _kp(p, "right_wrist")

        if ls and rs:
            sh_x[f] = (ls[0] + rs[0]) / 2.0
            sh_y[f] = (ls[1] + rs[1]) / 2.0
        if nose:
            head_y[f] = nose[1]
        if lh and rh:
            hip_y[f] = (lh[1] + rh[1]) / 2.0

        # 头顶：取鼻/眼/耳里最靠上的 y，再上抬一点当作真正头顶
        head_pts = [q for q in (nose, leye, reye, lear, rear) if q is not None]
        head_top = min(q[1] for q in head_pts) if head_pts else None

        # ---- ★头胸核心点集：头（鼻/眼/耳）+ 双肩（胸线用肩到肩下一段近似）----
        #      这是最高优先级，必须落在画面中央区域、绝不出画。
        core_pts = [nose, leye, reye, lear, rear, ls, rs]
        cx0, cx1, cy0, cy1 = _minmax(core_pts)
        if cx0 is not None:
            hc_x0[f], hc_x1[f] = cx0, cx1
            hc_y0[f] = head_top if head_top is not None else cy0
            # 胸线：肩往下延伸约半个肩宽，纳入上胸，避免只保头不保胸
            if ls and rs:
                shoulder_w = abs(ls[0] - rs[0])
                chest_y = max(ls[1], rs[1]) + 0.5 * shoulder_w
            else:
                chest_y = cy1
            hc_y1[f] = max(cy1, chest_y)

        # ---- 上半身安全点集：头顶 + 双肩 + 双髋（+ 手腕，避免举手被裁）----
        up_pts = [nose, leye, reye, lear, rear, ls, rs, lh, rh, lw, rw]
        ux0, ux1, uy0, uy1 = _minmax(up_pts)
        if ux0 is not None:
            up_x0[f], up_x1[f] = ux0, ux1
            up_y0[f] = head_top if head_top is not None else uy0
            up_y1[f] = max(uy1, hip_y[f]) if np.isfinite(hip_y[f]) else uy1

        # ---- 全身安全点集：在上半身基础上加膝、踝 ----
        fb_pts = up_pts + [lk, rk, la, ra]
        fx0, fx1, fy0, fy1 = _minmax(fb_pts)
        if fx0 is not None:
            fb_x0[f], fb_x1[f] = fx0, fx1
            fb_y0[f] = head_top if head_top is not None else fy0
            fb_y1[f] = fy1

    geom = {"bx": _interp_missing(bx), "by": _interp_missing(by),
            "bw": _interp_missing(bw), "bh": _interp_missing(bh),
            "head_y": head_y, "sh_x": sh_x, "sh_y": sh_y, "hip_y": hip_y,
            # 安全框极值：保留 NaN，装配阶段按需插值/回退
            "hc_x0": hc_x0, "hc_x1": hc_x1, "hc_y0": hc_y0, "hc_y1": hc_y1,
            "up_x0": up_x0, "up_x1": up_x1, "up_y0": up_y0, "up_y1": up_y1,
            "fb_x0": fb_x0, "fb_x1": fb_x1, "fb_y0": fb_y0, "fb_y1": fb_y1}

    # 无任何主体：给个居中假设，退化成温和运镜
    if np.allclose(geom["bw"], 0):
        geom["bx"][:] = src_w / 2.0; geom["by"][:] = src_h / 2.0
        geom["bw"][:] = src_w * 0.35; geom["bh"][:] = src_h * 0.75

    # ===================================================================
    # ★安全框极值先"保留全 NaN 段"地补齐短缺口，再做零相位轻平滑。
    #   这是硬约束逐帧生效的输入源；如果它本身逐帧跳动（关键点检测置信度
    #   抖动、遮挡、刚进画面时姿态不稳定等），后面"每帧都执行"的硬校正
    #   （见 _apply_safe_frame / _smooth_track 的平滑后硬校验）就会照单
    #   全收，直接体现成画面抖动——尤其片头主体刚入镜、检测最不稳的时候。
    # ===================================================================
    fps = float(ctx.timeline.fps)
    for k in ("hc_x0", "hc_x1", "hc_y0", "hc_y1",
              "up_x0", "up_x1", "up_y0", "up_y1",
              "fb_x0", "fb_x1", "fb_y0", "fb_y1",
              "sh_x", "sh_y", "head_y", "hip_y"):
        geom[k] = _light_smooth(_interp_missing_keep_allnan(geom[k]), fps)
    return geom


# ===================================================================
# ★新增：取某帧的安全框（源坐标 x0,y0,x1,y1）。全身优先关键点，缺则回退 bbox。
# ===================================================================
def _safe_box(geom, i, kind, src_w, src_h, margin_frac):
    """
    返回该帧某类"必须被裁剪窗包住"的矩形 (x0,y0,x1,y1) 及其中心。
    kind ∈ {"core", "upper", "full"}：
      - core ：头胸核心（头+双肩+上胸），★最高优先，必须在画面中央区域
      - upper：上半身（头顶+双肩+双髋+手腕）
      - full ：全身（再加膝+踝）
    关键点缺失时回退到 bbox 的相应区域。
    margin_frac：在关键点范围外再留的边距（占框宽/高的比例），避免贴边。
    """
    bx, by = geom["bx"][i], geom["by"][i]
    bw, bh = geom["bw"][i], geom["bh"][i]
    b_x0 = bx - bw / 2.0; b_y0 = by - bh / 2.0
    b_x1 = bx + bw / 2.0; b_y1 = by + bh / 2.0

    if kind == "full":
        x0, x1 = geom["fb_x0"][i], geom["fb_x1"][i]
        y0, y1 = geom["fb_y0"][i], geom["fb_y1"][i]
        if not (np.isfinite(x0) and np.isfinite(x1)):
            x0, x1 = b_x0, b_x1
        if not (np.isfinite(y0) and np.isfinite(y1)):
            y0, y1 = b_y0, b_y1
    elif kind == "core":
        x0, x1 = geom["hc_x0"][i], geom["hc_x1"][i]
        y0, y1 = geom["hc_y0"][i], geom["hc_y1"][i]
        # 回退：bbox 上 30%（头到上胸大约在此）
        if not (np.isfinite(x0) and np.isfinite(x1)):
            x0, x1 = b_x0 + bw * 0.20, b_x1 - bw * 0.20
        if not (np.isfinite(y0) and np.isfinite(y1)):
            y0, y1 = b_y0, b_y0 + bh * 0.30
    else:  # upper
        x0, x1 = geom["up_x0"][i], geom["up_x1"][i]
        y0, y1 = geom["up_y0"][i], geom["up_y1"][i]
        if not (np.isfinite(x0) and np.isfinite(x1)):
            x0, x1 = b_x0, b_x1
        if not (np.isfinite(y0) and np.isfinite(y1)):
            y0, y1 = b_y0, b_y0 + bh * 0.60

    w = max(1.0, x1 - x0); h = max(1.0, y1 - y0)
    mx = margin_frac * w; my = margin_frac * h
    x0 -= mx; x1 += mx; y0 -= my; y1 += my
    # ★上边界抬到「真实颅顶」：关键点最高的是 nose/eye/ear（眉眼高度），
    #   比真实头顶低了大半个头。core/upper 是要"必须包住"的框，
    #   若上边界只到眉眼，头顶那一截就在保护范围之外 → 头顶留白被吃掉甚至切头。
    if kind in ("core", "upper"):
        crown = _crown_y(geom["head_y"][i], geom["sh_y"][i], bh, b_y0)
        if np.isfinite(crown):
            y0 = min(y0, crown - my)
    cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
    return x0, y0, x1, y1, cx, cy


# ===================================================================
# ★新增：脖子（双肩连线中点）近似位置——构图锚点应该钉在这里，
#   而不是"头胸核心框"的几何中心。核心框是"头顶→上胸"这一段，
#   它的几何中心天然比脖子更靠上（偏头部），如果拿框中心去对齐
#   构图锚点，脖子/上胸看起来就会比预期更靠上。
# ===================================================================
def _neck_point(geom, i, core_box):
    """
    优先用双肩中点 sh_x/sh_y 作为"脖子"位置；缺失（未检测到双肩）时，
    退回头胸核心框内约 65% 高度处（经验比例：头顶到肩略低于框中点）。
    """
    x0, y0, x1, y1 = core_box[:4]
    nx = geom["sh_x"][i]; ny = geom["sh_y"][i]
    if not (np.isfinite(nx) and np.isfinite(ny)):
        nx = (x0 + x1) / 2.0
        ny = y0 + 0.65 * (y1 - y0)
    return float(nx), float(ny)


# ===================================================================
# ★新增：按主体真实大小做景别可行性降级
# ===================================================================
_SHOT_ORDER = ["closeup", "medium", "wide", "extreme_wide"]  # 从近到远（统一新枚举）


def _downgrade_shots_by_subject_size(plan, geom, n, base_h, cam_cfg):
    """
    生成视频前的"景别体检"：逐段看主体在段内的最大高度占画面比例 r，
    如果当前景别在 r 下会导致头胸/上半身装不下，就把该段降级到更远的景别。
    这从源头避免"人很大却给 closeup/medium 导致下半身甚至上半身出画"。

    规则（可配 camera.safe_frame.downgrade_ratio）：
      r >= wide_at   -> 强制 wide
      r >= medium_at -> 最近只能 medium
      r >= upper_at  -> 最近只能 upper
      否则           -> 不限制（可 closeup）
    地面动作段（intensity 高的 level_change/freeze 已在别处处理）这里按几何统一体检。

    ★单位修正：bh（主体 bbox 高）是源像素坐标系下的量，必须除以同一坐标系
      下"zoom=1 时裁剪窗的高度"（即 base_h = largest_source_view 算出的
      base_h），而不是输出分辨率 out_h。两者单位不同——当源分辨率明显高于
      （或低于）输出分辨率时（例如 4K 源剪 1080p，或 demo 场景做等比放大），
      用 out_h 当分母会让 r 系统性偏小或偏大，导致降级规则几乎不触发，
      表现为"无论人多大，大多数时候都还是近景"。
    """
    sf = cam_cfg.get("safe_frame", {})
    if not bool(sf.get("enabled", True)):
        return plan
    dr = sf.get("downgrade_ratio", {})
    wide_at = float(dr.get("wide", 0.82))
    medium_at = float(dr.get("medium", 0.68))
    upper_at = float(dr.get("upper", 0.55))

    bh = geom["bh"]; by = geom["by"]
    changed = 0
    for seg in plan:
        s = max(0, int(seg["start_f"])); e = min(n, int(seg["end_f"]))
        if e <= s:
            continue
        # 段内主体最大"可见高度占比"：bbox 高 与 zoom=1 时裁剪窗高的比值，
        # 两者同为源像素坐标系，比值才有意义。
        r = float(np.nanmax(bh[s:e])) / float(base_h)
        # 该段允许的"最近景别"
        if r >= wide_at:
            allowed = "wide"
        elif r >= medium_at:
            allowed = "medium"
        elif r >= upper_at:
            allowed = "medium"   # 旧 upper 档并入 medium（新枚举无 upper）
        else:
            allowed = "closeup"
        cur = seg["shot"]
        # 若当前景别比允许的更近，则降级到 allowed
        if _SHOT_ORDER.index(cur) < _SHOT_ORDER.index(allowed):
            seg["_shot_orig"] = cur
            seg["shot"] = allowed
            changed += 1
    if changed:
        print(f"[camera] 景别体检：{changed} 段因主体过大被降级（保头胸+尽量全身）")
    return plan


def _secondary_kind(shot, want_fb):
    """
    ★按景别决定「次级安全框」要保住哪一段身体。

    这是"安全框按景别只保该保的"——否则无论什么景别都强制全身入画，
    中景(要切腿)、特写(要切到胸)在几何上永远不可能出现，全片只能是远景。

      · 地面动作帧 want_fb（level_change/freeze/floor…）：强制全身（安全兜底，最高优先）
      · wide / extreme_wide ：全身必须在（这本就是该景别的定义）
      · medium              ：只保上半身（允许切腿）
      · closeup             ：只保头胸核心（允许切到胸）→ 返回 None
    头胸核心 core 始终是硬约束，不受此函数影响。
    """
    if want_fb:
        return "full"
    if shot in ("wide", "extreme_wide"):
        return "full"
    if shot == "medium":
        return "upper"
    return None       # closeup：仅受 core 约束


def _apply_safe_frame_kinds(shot, want_fb):
    """给 _smooth_track / _apply_beat_accents 用的同一套判定（保持三处一致）。"""
    return _secondary_kind(shot, want_fb)


# ----------------------------- 景别 → 目标 zoom/center -----------------------------
def _crown_y(head_y, sh_y, bh, top):
    """
    估计**真实颅顶**的 y。
    ★COCO-17 最高的关键点是 nose/eye/ear，没有颅顶点：
      min(y) 拿到的是眉眼高度，比真实头顶低了大半个头。
      直接拿它当"头顶"来构图，头顶留白就会被吃掉大半（或头被切）。
    解剖比例：鼻→肩 ≈ 0.85 个头高，鼻→颅顶 ≈ 0.55 个头高
             → 颅顶 ≈ nose_y − 0.65 × (shoulder_y − nose_y)
    """
    if np.isfinite(head_y) and np.isfinite(sh_y) and sh_y > head_y:
        return head_y - 0.65 * (sh_y - head_y)
    if np.isfinite(head_y):
        return head_y - 0.06 * bh          # 无肩点时按身高粗估
    return top


def _shot_targets(shot, geom, i, base_h, cover_cfg, src_w, src_h):
    """给定景别，算这一帧的目标覆盖内容高度、目标缩放、目标中心锚点。"""
    bx, by = geom["bx"][i], geom["by"][i]
    bw, bh = geom["bw"][i], geom["bh"][i]
    head_y = geom["head_y"][i]; sh_x = geom["sh_x"][i]; sh_y = geom["sh_y"][i]
    hip_y = geom["hip_y"][i]

    top = by - bh / 2.0
    cover = float(cover_cfg.get(shot, 0.78))
    crown = _crown_y(head_y, sh_y, bh, top)      # ★真实颅顶，构图上边界用它

    if shot in ("wide", "extreme_wide"):
        content_h = bh
        cx_t, cy_t = bx, by
    elif shot == "medium":
        # 中景：颅顶 → 大腿中部（髋下方约半个"肩→髋"）。
        #   旧写法 content_h = bh*0.85 取"全身的85%"，等于把整个人塞进画面 → 顶天立地。
        torso = (hip_y - sh_y) if (np.isfinite(hip_y) and np.isfinite(sh_y)
                                   and hip_y > sh_y) else 0.28 * bh
        y_top = crown
        y_bot = (hip_y + 0.6 * torso) if np.isfinite(hip_y) else top + 0.70 * bh
        content_h = max(1.0, y_bot - y_top)
        cx_t = sh_x if np.isfinite(sh_x) else bx
        cy_t = (y_top + y_bot) / 2.0
    elif shot == "closeup":
        # ★特写规格（按需求定义，不再靠经验系数猜）：
        #     腰(髋) 落在画面下边框，肩落在画面水平中线。
        #   由此反解：肩在中线 → 视窗中心 cy = sh_y
        #             髋在下框 → view_h/2 = hip_y − cy  →  view_h = 2·(hip_y − sh_y)
        #   头顶留白因此自动得到 = view_h/2 − (sh_y − crown)，约占画面高 20%。
        if np.isfinite(sh_y) and np.isfinite(hip_y) and hip_y > sh_y:
            content_h = max(1.0, 2.0 * (hip_y - sh_y))
            cy_t = sh_y
        else:                                   # 关键点缺失时的保守回退
            y_ref = sh_y if np.isfinite(sh_y) else top + 0.20 * bh
            content_h = max(1.0, (y_ref - crown) * 2.4)
            cy_t = crown + content_h * 0.42
        cx_t = sh_x if np.isfinite(sh_x) else bx
    else:
        content_h = bh; cx_t, cy_t = bx, by

    # 覆盖率 cover = content_h / view_h，view_h = base_h/z  =>  z = cover*base_h/content_h
    z_t = cover * base_h / max(1.0, content_h)
    return z_t, cx_t, cy_t


# ----------------------------- 主流程 -----------------------------
@register("camera")
class CameraStage(Stage):
    name = "camera"

    def run(self, ctx):
        cam_cfg = ctx.config.get("camera", {})
        cam_type = cam_cfg.get("type", "rule")
        n = int(ctx.timeline.frame_count)
        fps = float(ctx.timeline.fps)
        src_w = int(ctx.meta["width"]); src_h = int(ctx.meta["height"])
        out_w = int(ctx.config["output"]["width"]); out_h = int(ctx.config["output"]["height"])
        base_w, base_h = largest_source_view(src_w, src_h, out_w, out_h)

        if cam_type == "passthrough":
            ctx.camera_track = [CameraParams(zoom=1.0, cx=src_w / 2.0, cy=src_h / 2.0)
                                for _ in range(n)]
            ctx.timeline.assert_track_length(len(ctx.camera_track))
            print(f"[camera] passthrough：{n} 帧恒等轨迹")
            return ctx
        if cam_type != "rule":
            raise NotImplementedError(f"camera.type '{cam_type}' 未实现")

        emax = effective_max_zoom(ctx.config, src_w, src_h, out_w, out_h)
        if emax <= 1.0001:
            print("[camera][提示] effective_max_zoom=1.0，运镜会被夹平。")
            print("             设 camera.allow_upscale_for_demo=true 或用更高分辨率源。")

        plan = ctx.extras.get("shot_plan") or [{"start_f": 0, "end_f": n,
                                               "shot": "medium", "move": "follow",
                                               "priority": 0, "src": "none", "intensity": 0.0}]
        music = ctx.extras.get("music", {}) or {}
        energy = np.array(music.get("energy_curve") or [0.0] * n, dtype=float)
        if len(energy) < n:
            energy = np.pad(energy, (0, n - len(energy)), mode="edge")
        energy = energy[:n]

        log("构建主体几何序列（关键点极值/安全框输入）…")
        geom = _subject_geometry(ctx, n)
        cover_cfg = cam_cfg.get("shot_coverage",
                                {"wide": 0.60, "medium": 0.78, "upper": 0.95, "closeup": 1.0,
                                 "extreme_wide": 0.45})
        # 注：closeup 的 cover 必须是 1.0 —— 它的 content_h 已按
        #     「腰在下框、肩在中线」精确反解出视窗高，再乘系数就破坏规格。

        # ★生成前景别体检：主体过大的段自动降级，从源头避免上半身出画
        #   注意：这里必须传 base_h（源坐标系下 zoom=1 的裁剪窗高），
        #   不能传 out_h（输出分辨率），两者不同单位，误用会让降级失效。
        plan = _downgrade_shots_by_subject_size(plan, geom, n, base_h, cam_cfg)

        # ★安全框：每帧是否要求全身出镜
        fullbody_mask = _fullbody_mask(ctx, n, fps, cam_cfg)

        # 每帧的景别/运动/段内进度
        shot_of = np.empty(n, dtype=object)
        move_of = np.empty(n, dtype=object)
        seg_u = np.zeros(n)           # 段内归一化进度 0..1
        seg_inten = np.zeros(n)
        for seg in plan:
            s, e = int(seg["start_f"]), int(seg["end_f"])
            e = min(e, n); s = max(0, s)
            L = max(1, e - s)
            for i in range(s, e):
                shot_of[i] = seg["shot"]; move_of[i] = seg["move"]
                seg_u[i] = (i - s) / L
                seg_inten[i] = float(seg.get("intensity", 0.0))
        for i in range(n):        # 兜底
            if shot_of[i] is None:
                shot_of[i] = "medium"; move_of[i] = "follow"

        # ---- 逐帧目标 zoom/center，叠加 move 包络 ----
        log("逐帧景别→目标 zoom/center + 安全框硬约束…")
        fcfg = cam_cfg.get("follow", {})
        rot_cfg = cam_cfg.get("rotation", {})
        rot_on = bool(rot_cfg.get("enabled", True))
        rot_max = float(rot_cfg.get("max_deg", 10))
        eg = float(_safe(ctx.config, "shotplan", "energy_gain", default=1.0))

        z_tgt = np.zeros(n); cx_tgt = np.zeros(n); cy_tgt = np.zeros(n); rot_tgt = np.zeros(n)
        z_want_dbg = np.zeros(n); z_safe_dbg = np.zeros(n)   # ★诊断：设计想要 vs 安全框允许
        for i in range(n):
            shot = shot_of[i]; move = move_of[i]
            z_t, cx_t, cy_t = _shot_targets(shot, geom, i, base_h, cover_cfg, src_w, src_h)
            u = seg_u[i]
            gain = clamp(eg * (0.4 + 0.9 * energy[i]), 0.2, 1.6)

            if move == "push_in":
                z_t = z_t * _lerp(0.92, 1.56, _smoothstep(u))
            elif move == "pull_out":
                z_t = z_t * _lerp(1.56, 0.90, _smoothstep(u))
            elif move == "orbit":
                amp = base_w * 0.06 * gain
                cx_t = cx_t + amp * math.sin(2 * math.pi * u)
                if rot_on:
                    rot_tgt[i] = 0.4 * rot_max * math.sin(2 * math.pi * u) * gain
            elif move == "roll" and rot_on:
                rot_tgt[i] = rot_max * math.sin(math.pi * u) * clamp(0.5 + seg_inten[i], 0.4, 1.0)
            # static / follow / recenter：不额外改 zoom（recenter 仅换锚点，已在 center 里）

            # ===========================================================
            # ★安全框硬约束（核心）：装不下就压低 zoom、把 center 拉向安全框
            # ===========================================================
            want_fb = bool(fullbody_mask[i])
            z_want_dbg[i] = z_t                       # 景别+运镜设计想要的 zoom（未受任何约束）
            z_t, cx_t, cy_t = self._apply_safe_frame(
                geom, i, want_fb, z_t, cx_t, cy_t,
                base_h, out_w, out_h, src_w, src_h, emax, cam_cfg, shot=shot)
            z_safe_dbg[i] = z_t                       # 安全框允许的 zoom

            z_tgt[i] = clamp(z_t, float(fcfg.get("min_zoom", 1.0)), emax)
            cx_tgt[i] = cx_t; cy_tgt[i] = cy_t

        # ★诊断：到底是谁不让推近——max_zoom（画质预算）还是安全框？
        # 注意：_apply_safe_frame 内部也把 z 夹到 emax，所以要和 min(want, emax) 比，
        # 才能把"安全框自己的限制"从"emax 的限制"里摘出来，否则会冤枉安全框。
        want_med = float(np.median(z_want_dbg)); want_max = float(np.max(z_want_dbg))
        want_capped = np.minimum(z_want_dbg, emax)
        sat_emax = float(np.mean(z_want_dbg > emax * 1.001))
        sat_safe = float(np.mean(z_safe_dbg < want_capped * 0.999))
        log(f"[诊断] 设计想要 zoom：中位={want_med:.2f} 最大={want_max:.2f} | 上限 emax={emax:.2f}")
        log(f"[诊断] {sat_emax*100:.0f}% 帧想推近却被 max_zoom 夹住 · "
            f"{sat_safe*100:.0f}% 帧被安全框额外压低（已扣除 emax 的影响）")
        if sat_emax > 0.5:
            log("[诊断][提示] 过半帧顶到 max_zoom：主体在源画面里太小，近景需要放大。"
                "  想要近景请设 camera.allow_upscale_for_demo=true 并调高 camera.max_zoom"
                f"（此片约需 {want_med:.1f}），代价是上采样掉画质。")

        log("平滑轨迹（One Euro + 死区 + 限速 + 平滑后硬校验）…")
        track = self._smooth_track(ctx, n, src_w, src_h, out_w, out_h, base_h, emax,
                                   z_tgt, cx_tgt, cy_tgt, rot_tgt, shot_of, move_of, fcfg,
                                   geom, fullbody_mask)

        # ---- 拍点修饰层（zoom 叠加）----
        keyframes = self._plan_beat_keyframes(ctx, music, n, energy, shot_of, move_of)
        track = self._apply_beat_accents(ctx, track, keyframes, emax,
                                         geom, fullbody_mask, src_w, src_h,
                                         out_w, out_h, base_h, cam_cfg)

        ctx.camera_track = track
        ctx.timeline.assert_track_length(len(track))

        # ---- 导出 ----
        out_dir = ctx.extras.get("analysis_out_dir", ".")
        os.makedirs(out_dir, exist_ok=True)
        track_json = [{"zoom": round(float(c.zoom), 6),
                       "cx": round(float(c.cx), 3), "cy": round(float(c.cy), 3),
                       "rot": round(float(c.rot), 4), "blur": round(float(c.blur), 4),
                       "move": c.move, "shot": c.shot} for c in track]
        metrics = self._metrics(ctx, track, out_w, out_h, keyframes, geom, fullbody_mask)
        for name, data in (("camera_track.json", track_json),
                           ("camera_keyframes.json", keyframes),
                           ("metrics.json", metrics)):
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[camera] rule：{len(plan)} 段分镜 + {len(keyframes)} 个拍点修饰, "
              f"effective_max_zoom={emax:.2f}, 全身约束帧={int(fullbody_mask.sum())}/{n}")
        print(f"[camera.metrics] {metrics}")
        return ctx

    # ===============================================================
    # ★新增：安全框硬约束的核心计算
    # ===============================================================
    def _max_zoom_for_box(self, box_w, box_h, base_h, out_w, out_h):
        """
        给定必须包住的安全框尺寸（源像素），算出"仍能把它整个装进裁剪窗"的最大 zoom。
        裁剪窗尺寸：view_w = base_h*(out_w/out_h)/z, view_h = base_h/z。
        要求 view_w >= box_w 且 view_h >= box_h：
            z <= base_h*(out_w/out_h)/box_w  且  z <= base_h/box_h
        """
        view_w_at_1 = base_h * (out_w / out_h)   # z=1 时窗宽
        view_h_at_1 = base_h                     # z=1 时窗高
        z_by_w = view_w_at_1 / max(1.0, box_w)
        z_by_h = view_h_at_1 / max(1.0, box_h)
        return min(z_by_w, z_by_h)

    def _fit_box_in_view(self, box, view_w, view_h, cx_t, cy_t, center_pull=0.0,
                        x_anchor_frac=0.5, y_anchor_frac=0.5, anchor_point=None):
        """
        给定必须包住的框 box=(x0,y0,x1,y1) 和窗尺寸，返回把框收进窗的合法 center。
        center_pull∈[0,1]：把 center 额外拉向"构图锚点"的比例（0=只夹到合法
        区间，1=直接对齐锚点）。

        x_anchor_frac / y_anchor_frac：锚点在裁剪窗内的目标相对位置
        （0=贴窗左/上边，0.5=正中，1=贴窗右/下边）。默认 0.5 即严格居中；
        想要"偏画面中上、黄金分割线附近"就传 y_anchor_frac≈0.35~0.42
        （小于 0.5，因为要往上移）。

        anchor_point：可选 (ax, ay)，指定"构图上真正想对齐的那个点"
        （比如脖子/双肩中点），与 box 解耦——box 只负责"必须整个装进
        画面"这个硬约束，anchor_point 才是"往哪个位置摆"的构图诉求。
        不传则退化为用 box 自己的几何中心当锚点（原行为）。
        换算：若想让锚点落在窗内 anchor 位置，则 camera 中心
        （= 窗几何中心在源坐标系里的位置）要相应偏移：
            cam_center = anchor_point + view_size * (0.5 - anchor_frac)

        窗装不下框时，直接对齐框几何中心（损失最小，此时构图诉求让位给
        "至少要包住"这个更硬的约束）。
        """
        x0, y0, x1, y1 = box
        bcx = (x0 + x1) / 2.0; bcy = (y0 + y1) / 2.0
        box_w = x1 - x0; box_h = y1 - y0
        acx, acy = anchor_point if anchor_point is not None else (bcx, bcy)
        tcx = acx + view_w * (0.5 - x_anchor_frac)
        tcy = acy + view_h * (0.5 - y_anchor_frac)

        if view_w >= box_w:
            lo = x1 - view_w / 2.0; hi = x0 + view_w / 2.0
            cx = clamp(cx_t, lo, hi)
            cx = cx + (tcx - cx) * center_pull      # 额外向构图锚点拉
            cx = clamp(cx, lo, hi)
        else:
            cx = bcx
        if view_h >= box_h:
            lo = y1 - view_h / 2.0; hi = y0 + view_h / 2.0
            cy = clamp(cy_t, lo, hi)
            cy = cy + (tcy - cy) * center_pull
            cy = clamp(cy, lo, hi)
        else:
            cy = bcy
        return cx, cy

    def _apply_safe_frame(self, geom, i, want_fb, z_t, cx_t, cy_t,
                          base_h, out_w, out_h, src_w, src_h, emax, cam_cfg,
                          shot="wide"):
        """
        ★分层安全约束，优先级从高到低：头胸核心 > (全身 | 上半身)。
          1) 头胸核心框：最高优先。zoom 不得大于"能装下核心框"的上限；
             且 center 额外向核心框中心拉近（core_center_pull），保证头胸靠画面中央。
          2) 次级框：地面动作段用全身框，否则上半身框。在不牺牲核心的前提下，
             进一步压低 zoom 以尽量把次级框也纳入（"尽最大努力全包"）。
          3) center 最终以"能同时容纳核心框"为硬底线做夹取。
        "包不下就顶边界不再拉远"：所有 z_fit 都 clamp 到 [1.0, emax]。
        """
        sf_cfg = cam_cfg.get("safe_frame", {})
        if not bool(sf_cfg.get("enabled", True)):
            return z_t, cx_t, cy_t

        margin = float(sf_cfg.get("margin_frac", 0.06))
        core_margin = float(sf_cfg.get("core_margin_frac", 0.10))   # 核心框留白更多→更居中
        core_pull = float(sf_cfg.get("core_center_pull", 0.6))      # 核心居中强度
        # ★构图锚点：脖子（双肩连线中点）放在画面中上、标准黄金分割线处
        #   （黄金分割 1-0.618=0.382，即从顶部往下 38.2% 的位置）。
        #   不是把"头胸核心框的几何中心"钉在这条线上——那个几何中心天然
        #   偏头部（框是"头顶→上胸"，中点比脖子更靠上），钉在这里会让
        #   整体看起来比预期更靠上。
        core_y_anchor = float(sf_cfg.get("core_y_anchor", 0.45))
        # print(f"[debug] core_y_anchor={core_y_anchor}")

        # --- 头胸核心框（最高优先）---
        cx0, cy0, cx1, cy1, ccx, ccy = _safe_box(geom, i, "core", src_w, src_h, core_margin)
        core_w = cx1 - cx0; core_h = cy1 - cy0
        z_core = clamp(self._max_zoom_for_box(core_w, core_h, base_h, out_w, out_h), 1.0, emax)

        # --- 次级框：按景别只保该保的那一段（见 _secondary_kind）---
        kind = _secondary_kind(shot, want_fb)
        fx0, fy0, fx1, fy1, _, _ = _safe_box(geom, i, "full", src_w, src_h, margin)
        z_full = clamp(self._max_zoom_for_box(fx1 - fx0, fy1 - fy0, base_h, out_w, out_h), 1.0, emax)
        ux0, uy0, ux1, uy1, _, _ = _safe_box(geom, i, "upper", src_w, src_h, margin)
        z_upper = clamp(self._max_zoom_for_box(ux1 - ux0, uy1 - uy0, base_h, out_w, out_h), 1.0, emax)

        # zoom 约束：core 永远是硬上限；次级按景别决定
        #   full  → 全身+上半身都必须在（远景/大远景/地面动作）
        #   upper → 只保上半身，允许切腿（中景）
        #   None  → 只保头胸核心，允许切到胸（特写）
        z_new = min(z_t, z_core)                       # 头胸核心：硬上限，绝不突破
        if kind == "full":
            z_new = min(z_new, z_upper, z_full)
            sx0, sy0, sx1, sy1 = fx0, fy0, fx1, fy1    # center 以全身框为准
        elif kind == "upper":
            z_new = min(z_new, z_upper)
            sx0, sy0, sx1, sy1 = ux0, uy0, ux1, uy1    # center 以上半身框为准
        else:                                          # closeup
            sx0, sy0, sx1, sy1 = cx0, cy0, cx1, cy1    # center 以核心框为准
        z_new = max(z_new, 1.0)

        # 次级框仍装不下时退回上半身框做 center 基准，避免顶飞
        view_h_try = base_h / max(z_new, 1.0)
        if kind == "full" and (fy1 - fy0) > view_h_try:
            sx0, sy0, sx1, sy1 = ux0, uy0, ux1, uy1

        view_w = base_h * (out_w / out_h) / z_new
        view_h = base_h / z_new

        # ★构图锚点用脖子（双肩中点），"必须包住"仍然用核心框——两者解耦。
        neck_pt = _neck_point(geom, i, (cx0, cy0, cx1, cy1))

        # center：先按次级框夹到合法区间（尽量全包），
        #         再用核心框做硬底线夹取 + 往脖子锚点拉近（核心优先级更高，后作用覆盖）。
        cx_new, cy_new = self._fit_box_in_view(
            (sx0, sy0, sx1, sy1), view_w, view_h, cx_t, cy_t, center_pull=0.0)
        cx_new, cy_new = self._fit_box_in_view(
            (cx0, cy0, cx1, cy1), view_w, view_h, cx_new, cy_new,
            center_pull=core_pull, y_anchor_frac=core_y_anchor, anchor_point=neck_pt)

        return z_new, cx_new, cy_new

    # ------------------------- 平滑轨迹 -------------------------
    def _smooth_track(self, ctx, n, src_w, src_h, out_w, out_h, base_h, emax,
                      z_tgt, cx_tgt, cy_tgt, rot_tgt, shot_of, move_of, fcfg,
                      geom, fullbody_mask):
        fps = float(ctx.timeline.fps)
        fx = OneEuro(fps, fcfg.get("min_cutoff", 0.45), fcfg.get("beta", 0.03))
        fy = OneEuro(fps, fcfg.get("min_cutoff", 0.45), fcfg.get("beta", 0.03))
        fz = OneEuro(fps, fcfg.get("zoom_min_cutoff", 0.30), fcfg.get("zoom_beta", 0.015))
        fr = OneEuro(fps, fcfg.get("rot_min_cutoff", 0.8), fcfg.get("rot_beta", 0.02))

        dz_x = float(fcfg.get("deadzone_x", 0.10))
        dz_y = float(fcfg.get("deadzone_y", 0.12))
        max_step = float(fcfg.get("max_center_step_px", 32.0))

        sf_cfg = ctx.config.get("camera", {}).get("safe_frame", {})
        sf_on = bool(sf_cfg.get("enabled", True))
        sf_margin = float(sf_cfg.get("margin_frac", 0.06))
        # 平滑后再校验一次的收紧边距：略小于目标边距，允许平滑吃掉一点余量但不吃到人身上
        sf_hard_margin = float(sf_cfg.get("hard_margin_frac", 0.0))

        # ★片头防抖：cx_prev/cy_prev 不再假设"从画面正中心开始"，而是直接用
        #   第 0 帧的真实目标（已经过 _apply_safe_frame 处理，含构图锚点）。
        #   否则如果主体一开始就不在画面中心，死区+限速会从"画面中心"往
        #   "主体实际位置"缓慢爬升，同时 One Euro 在首次遇到大位移时的
        #   速度估计也会不准，两者叠加就是开头那几秒的明显晃动/漂移。
        cx_prev = float(cx_tgt[0]) if n > 0 else src_w / 2.0
        cy_prev = float(cy_tgt[0]) if n > 0 else src_h / 2.0

        # 注：构图锚点（往中上/黄金分割拉）只在 _apply_safe_frame 算"平滑
        #   目标"时生效一次，之后正常走 One Euro+死区+限速。下面这段循环里
        #   的硬校验不再重复"往锚点拉"，只做安全区间夹取 + 限速，见循环内注释。

        # ★硬切帧集合：这些帧让裁剪窗"瞬间跳"到新构图。
        #   裁剪相机没有真多机位，"切"就是构图瞬变；不这样做的话
        #   One Euro + 限速(max_center_step_px)会把景别切换抹成一段缓慢推移
        #   —— 表现为"全是移动运镜、一个硬切也没有"。
        cut_at = set()
        for seg in (ctx.extras.get("shot_plan") or []):
            if seg.get("cut"):
                cut_at.add(int(seg["start_f"]))
        if cut_at:
            log(f"硬切 {len(cut_at)} 处：这些帧裁剪窗瞬间跳转（不走平滑/限速）")

        track = []
        _prog = Progress(n, "平滑", every_frac=0.1, min_step=200)
        for i in range(n):
            _prog.update(i)
            if i in cut_at:
                # 硬切：清掉滤波状态、把 prev 直接置为本帧目标 →
                # 本帧输出即目标构图，且不受死区/限速约束。
                fx.reset(); fy.reset(); fz.reset(); fr.reset()
                cx_prev, cy_prev = float(cx_tgt[i]), float(cy_tgt[i])
            z = clamp(fz(z_tgt[i]), 1.0, emax)
            view_w = base_h * (out_w / out_h) / max(z, 1.0)
            view_h = base_h / max(z, 1.0)

            # 死区：目标在安全区内则中心不动
            dx = cx_tgt[i] - cx_prev; dy = cy_tgt[i] - cy_prev
            lim_x = dz_x * view_w; lim_y = dz_y * view_h
            cx_des = cx_tgt[i] - math.copysign(lim_x, dx) if abs(dx) > lim_x else cx_prev
            cy_des = cy_tgt[i] - math.copysign(lim_y, dy) if abs(dy) > lim_y else cy_prev

            cx_s = fx(cx_des); cy_s = fy(cy_des)
            cx = cx_prev + clamp(cx_s - cx_prev, -max_step, max_step)
            cy = cy_prev + clamp(cy_s - cy_prev, -max_step, max_step)

            # ===========================================================
            # ★平滑后硬校验：One Euro/死区/限速可能又把安全框顶出去，
            #   这里对 zoom 和 center 做最后一道"不可协商"的纠正。
            #
            # ★关键修正：构图锚点（往中上/黄金分割拉）只应该在"算平滑目标"
            #   那一步（_apply_safe_frame）生效一次，让它随后正常走 One Euro+
            #   死区+限速被平滑掉。这里的硬校验只是兜底"别把人挤出画面"的
            #   安全网，不应该再用 core_center_pull 主动把镜头往锚点拉一次——
            #   之前的写法是每一帧都无条件拉 60%，直接绕开了限速，只要姿态
            #   稍快（转身/伸展）或 zoom 被拍点顶到上限，这一拉就会在一帧内
            #   把镜头"焊"到新位置，看起来就是突然居中一下。
            #   现在改成：只做安全区间夹取（center_pull=0，不主动居中），
            #   并且这次纠偏本身也过一次限速，绝不允许单帧大跳。
            # ===========================================================
            if sf_on:
                want_fb = bool(fullbody_mask[i])
                # 核心框（硬）
                cx0, cy0, cx1, cy1, ccx, ccy = _safe_box(
                    geom, i, "core", src_w, src_h, sf_hard_margin)
                cw = cx1 - cx0; ch = cy1 - cy0
                z_core = clamp(self._max_zoom_for_box(cw, ch, base_h, out_w, out_h), 1.0, emax)
                # 次级框（软，尽量全包）
                # ★次级框按景别选（与 _apply_safe_frame 同一套判定），
                #   否则中景/特写会被"全身必须在"重新压回远景。
                _kind = _apply_safe_frame_kinds(str(shot_of[i]), want_fb)
                sx0, sy0, sx1, sy1, scx, scy = _safe_box(
                    geom, i, _kind or "core", src_w, src_h, sf_hard_margin)
                sw = sx1 - sx0; sh_ = sy1 - sy0
                z_sec = clamp(self._max_zoom_for_box(sw, sh_, base_h, out_w, out_h), 1.0, emax)

                z_lim = min(z_core, z_sec)
                if z > z_lim:                     # 平滑把 zoom 抬过上限 → 压回
                    z = z_lim
                view_w = base_h * (out_w / out_h) / max(z, 1.0)
                view_h = base_h / max(z, 1.0)
                # 只做安全区间夹取（不主动居中构图），并对纠偏量限速
                cx_fit, cy_fit = self._fit_box_in_view((sx0, sy0, sx1, sy1),
                                                       view_w, view_h, cx, cy, center_pull=0.0)
                cx_fit, cy_fit = self._fit_box_in_view((cx0, cy0, cx1, cy1),
                                                       view_w, view_h, cx_fit, cy_fit, center_pull=0.0)
                cx = cx + clamp(cx_fit - cx, -max_step, max_step)
                cy = cy + clamp(cy_fit - cy, -max_step, max_step)

            x0, y0, vw, vh, z = crop_rect(src_w, src_h, out_w, out_h, cx, cy, z)
            cx = x0 + vw / 2.0; cy = y0 + vh / 2.0
            rot = fr(rot_tgt[i])

            cam = CameraParams(zoom=float(z), cx=float(cx), cy=float(cy))
            cam.rot = float(rot); cam.blur = 0.0
            cam.move = str(move_of[i]); cam.shot = str(shot_of[i])
            track.append(cam)
            cx_prev, cy_prev = cx, cy
        return track

    # ------------------------- 拍点修饰 -------------------------
    def _plan_beat_keyframes(self, ctx, music, n, energy, shot_of, move_of):
        tcfg = ctx.config.get("camera", {}).get("templates", {})
        beat_cfg = tcfg.get("beat_pulse", {})
        punch_cfg = tcfg.get("downbeat_punch", {})
        beat_on = bool(beat_cfg.get("enabled", True))
        punch_on = bool(punch_cfg.get("enabled", True))

        keyframes = []
        for b in music.get("beat_grid", []):
            f = int(b.get("frame", -1))
            if not (0 <= f < n):
                continue
            strength = float(b.get("strength", 0.5) or 0.0)
            is_down = bool(b.get("is_downbeat", False))
            # freeze/closeup 段抑制大脉冲：只留极轻呼吸
            gate = 0.35 if (str(shot_of[f]) == "closeup" or str(move_of[f]) in ("push_in",)) else 1.0
            e_gain = 0.6 + 0.8 * energy[f]

            if beat_on:
                dz = float(beat_cfg.get("delta_zoom", 0.045)) * (0.7 + 0.5 * strength) * gate * e_gain
                keyframes.append({"frame": f, "move": "beat_pulse", "delta_zoom": round(dz, 5),
                                  "pre_f": int(beat_cfg.get("pre_f", 2)),
                                  "post_f": int(beat_cfg.get("post_f", 4)),
                                  "src": "beat", "strength": round(strength, 4)})
            if is_down and punch_on:
                dz = float(punch_cfg.get("delta_zoom", 0.14)) * (0.6 + 0.7 * strength) * gate * e_gain
                keyframes.append({"frame": f, "move": "downbeat_punch", "delta_zoom": round(dz, 10),
                                  "attack_f": int(punch_cfg.get("attack_f", 1)),
                                  "release_f": int(punch_cfg.get("release_f", 10)),
                                  "src": "downbeat", "strength": round(strength, 4)})

        keyframes.sort(key=lambda x: (x["frame"], x["move"]))
        return keyframes

    def _apply_beat_accents(self, ctx, track, keyframes, emax,
                            geom, fullbody_mask, src_w, src_h,
                            out_w, out_h, base_h, cam_cfg):
        n = len(track)
        zoom_add = np.zeros(n); blur_add = np.zeros(n)
        for kf in keyframes:
            f = int(kf["frame"]); dz = float(kf.get("delta_zoom", 0.0)); move = kf["move"]
            if move == "beat_pulse":
                pre = max(1, int(kf.get("pre_f", 2))); post = max(1, int(kf.get("post_f", 4)))
                s = max(0, f - pre); e = min(n - 1, f + post)
                for i in range(s, e + 1):
                    u = (i - s) / max(1, f - s) if i <= f else (i - f) / max(1, e - f)
                    shape = _smoothstep(u) if i <= f else 1.0 - _smoothstep(u)
                    zoom_add[i] += dz * shape
            elif move == "downbeat_punch":
                atk = max(1, int(kf.get("attack_f", 2))); rel = max(1, int(kf.get("release_f", 10)))
                s = max(0, f - atk); e = min(n - 1, f + rel)
                for i in range(s, e + 1):
                    u = (i - s) / max(1, f - s) if i <= f else (i - f) / max(1, e - f)
                    shape = _smoothstep(u) if i <= f else (1.0 - _smoothstep(u)) ** 1.2
                    zoom_add[i] += dz * shape
                    blur_add[i] += 0.22 * shape

        cap = float(_safe(ctx.config, "camera", "templates", "max_additive_zoom", default=0.84))
        zoom_add = np.clip(zoom_add, 0.0, cap)

        # ★拍点脉冲是"放大"（zoom+），会缩小裁剪窗、可能把安全框顶出去。
        #   叠加后对每帧再算一次安全框允许的 zoom 上限，把 pulse 削到不越界。
        #
        # ★关键修正：这里以前会用 core_center_pull 主动把镜头往构图锚点拉
        #   一次，而拍点导致 zoom 突然变化（裁剪窗突然变小/变大），"锚点"
        #   本身的换算结果也会跟着突变，每次拍点都在视觉上表现为"猛地居中
        #   一下"。构图锚点只该在算平滑目标时起一次作用，这里只负责"别把
        #   人挤出画面"的安全夹取（center_pull=0），并对纠偏量限速，
        #   跟主运镜用同一个 max_center_step_px 预算，绝不允许单帧大跳。
        sf_cfg = cam_cfg.get("safe_frame", {})
        sf_on = bool(sf_cfg.get("enabled", True))
        sf_hard_margin = float(sf_cfg.get("hard_margin_frac", 0.0))
        max_step = float(cam_cfg.get("follow", {}).get("max_center_step_px", 32.0))

        for i, cam in enumerate(track):
            z_new = clamp(float(cam.zoom) + float(zoom_add[i]), 1.0, float(emax))
            if sf_on:
                want_fb = bool(fullbody_mask[i])
                cx0, cy0, cx1, cy1, ccx, ccy = _safe_box(
                    geom, i, "core", src_w, src_h, sf_hard_margin)
                cw = cx1 - cx0; ch = cy1 - cy0
                z_core = clamp(self._max_zoom_for_box(cw, ch, base_h, out_w, out_h), 1.0, float(emax))
                _kind = _apply_safe_frame_kinds(str(cam.shot), want_fb)
                sx0, sy0, sx1, sy1, scx, scy = _safe_box(
                    geom, i, _kind or "core", src_w, src_h, sf_hard_margin)
                sw = sx1 - sx0; sh_ = sy1 - sy0
                z_sec = clamp(self._max_zoom_for_box(sw, sh_, base_h, out_w, out_h), 1.0, float(emax))

                z_lim = min(z_core, z_sec)
                if z_new > z_lim:                 # pulse 放大越过上限 → 削回
                    z_new = z_lim
                view_w = base_h * (out_w / out_h) / max(z_new, 1.0)
                view_h = base_h / max(z_new, 1.0)
                cx_fit, cy_fit = self._fit_box_in_view((sx0, sy0, sx1, sy1),
                                                       view_w, view_h, cam.cx, cam.cy, center_pull=0.0)
                cx_fit, cy_fit = self._fit_box_in_view((cx0, cy0, cx1, cy1),
                                                       view_w, view_h, cx_fit, cy_fit, center_pull=0.0)
                cx = cam.cx + clamp(cx_fit - cam.cx, -max_step, max_step)
                cy = cam.cy + clamp(cy_fit - cam.cy, -max_step, max_step)
                x0, y0, vw, vh, z_new = crop_rect(src_w, src_h, out_w, out_h, cx, cy, z_new)
                cam.cx = float(x0 + vw / 2.0); cam.cy = float(y0 + vh / 2.0)
            cam.zoom = float(z_new)
            cam.blur = float(max(cam.blur, blur_add[i]))
        return track

    # ------------------------- 指标 -------------------------
    def _metrics(self, ctx, track, out_w, out_h, keyframes, geom, fullbody_mask):
        src_w = int(ctx.meta["width"]); src_h = int(ctx.meta["height"])
        pose = ctx.extras.get("pose", {}) or {}
        boxes = {int(pf["frame"]): list(map(float, pf["bbox"]))
                 for pf in pose.get("pose_frames", []) if pf.get("bbox")}
        inside = []
        # ★命中率：头胸核心、上半身、全身段的全身；核心居中度
        core_in = []; upper_in = []; fullbody_in = []; core_centered = []
        for i, cam in enumerate(track):
            x0, y0, vw, vh, _ = crop_rect(src_w, src_h, out_w, out_h, cam.cx, cam.cy, cam.zoom)
            pad = 2.0
            b = boxes.get(i)
            if b:
                x, y, w, h = b
                ok = (x >= x0 - pad and y >= y0 - pad and
                      x + w <= x0 + vw + pad and y + h <= y0 + vh + pad)
                inside.append(1.0 if ok else 0.0)

            # ★头胸核心命中（最高优先，应接近 1.0）
            hx0, hy0, hx1, hy1, hccx, hccy = _safe_box(geom, i, "core", src_w, src_h, 0.0)
            core_ok = (hx0 >= x0 - pad and hy0 >= y0 - pad and
                       hx1 <= x0 + vw + pad and hy1 <= y0 + vh + pad)
            core_in.append(1.0 if core_ok else 0.0)
            # 核心居中度：核心框中心离画面中心的归一化距离（越小越居中）
            fcx = x0 + vw / 2.0; fcy = y0 + vh / 2.0
            core_centered.append(math.hypot((hccx - fcx) / max(1.0, vw),
                                            (hccy - fcy) / max(1.0, vh)))

            # 上半身安全框命中
            ux0, uy0, ux1, uy1, _, _ = _safe_box(geom, i, "upper", src_w, src_h, 0.0)
            up_ok = (ux0 >= x0 - pad and uy0 >= y0 - pad and
                     ux1 <= x0 + vw + pad and uy1 <= y0 + vh + pad)
            upper_in.append(1.0 if up_ok else 0.0)

            # 全身段的全身命中
            if bool(fullbody_mask[i]):
                fx0, fy0, fx1, fy1, _, _ = _safe_box(geom, i, "full", src_w, src_h, 0.0)
                fb_ok = (fx0 >= x0 - pad and fy0 >= y0 - pad and
                         fx1 <= x0 + vw + pad and fy1 <= y0 + vh + pad)
                fullbody_in.append(1.0 if fb_ok else 0.0)

        zs = np.array([c.zoom for c in track]); cxs = np.array([c.cx for c in track])
        cys = np.array([c.cy for c in track]); rots = np.array([c.rot for c in track])

        def jerk95(a):
            if len(a) < 4:
                return 0.0
            j = np.diff(a, n=3)
            return float(np.percentile(np.abs(j), 95)) if len(j) else 0.0

        return {"subject_in_frame_rate": round(float(np.mean(inside)) if inside else 0.0, 4),
                "head_chest_in_rate": round(float(np.mean(core_in)) if core_in else 1.0, 4),
                "head_chest_offcenter_mean": round(
                    float(np.mean(core_centered)) if core_centered else 0.0, 4),
                "upper_body_in_rate": round(float(np.mean(upper_in)) if upper_in else 1.0, 4),
                "fullbody_in_rate_on_masked": round(
                    float(np.mean(fullbody_in)) if fullbody_in else 1.0, 4),
                "fullbody_masked_frames": int(fullbody_mask.sum()),
                "zoom_min": round(float(zs.min()), 4), "zoom_max": round(float(zs.max()), 4),
                "rot_max_deg": round(float(np.abs(rots).max()), 3),
                "center_jerk95_px": round(max(jerk95(cxs), jerk95(cys)), 4),
                "zoom_jerk95": round(jerk95(zs), 6),
                "frames": len(track), "keyframes": len(keyframes)}
