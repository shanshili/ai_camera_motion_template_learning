"""
可选构图补丁 · 让景别留白（头顶1/4、脚下1/6、取景段、extreme_wide）真正生效
========================================================================
现状：工程里的 camera.py 是你的原版（安全核未动），景别留白只是近似居中。
本补丁把 camera._shot_targets 换成"模板驱动"版：读取模板注入的 shot_composition
（每档 {cover, span, head_top_frac, foot_bot_frac}），按 head_top_frac 精确控制
头顶留白，按 span 决定取景段（full/upper/chest），并正确处理 extreme_wide。

如何应用（二选一）：
  A. 手动：用下面的 _shot_targets 覆盖 pipeline_3/stages/camera.py 里的同名函数，
     并把 _SHOT_ORDER 改成 ["closeup","medium","wide","extreme_wide"]，
     把 _downgrade_shots_by_subject_size 里 allowed="upper" 改成 "medium"。
  B. 保持现状：不应用也能跑，extreme_wide 会退化成全身 wide，留白用近似值。

注意：本函数从 cam_cfg["shot_composition"]（模板 merge_into_config 注入）取留白；
     若拿不到则回退到 cover_cfg 的扁平覆盖率，行为等同原版。
"""

import numpy as np


def _shot_targets(shot, geom, i, base_h, cover_cfg, src_w, src_h, composition=None):
    """
    景别 -> 目标覆盖高/缩放/中心。
    composition: dict[shot] -> {cover, span, head_top_frac, foot_bot_frac}
                 （由模板注入 cam_cfg["shot_composition"]；缺则用 cover_cfg 扁平值）
    """
    bx, by = geom["bx"][i], geom["by"][i]
    bw, bh = geom["bw"][i], geom["bh"][i]
    head_y = geom["head_y"][i]; sh_x = geom["sh_x"][i]; sh_y = geom["sh_y"][i]
    hip_y = geom["hip_y"][i]
    top = by - bh / 2.0

    spec = (composition or {}).get(shot)
    if not isinstance(spec, dict):
        # 回退：只有扁平覆盖率，行为等同原版 wide/medium 处理
        cover = float(cover_cfg.get(shot, 0.78)) if not isinstance(
            cover_cfg.get(shot), dict) else float(cover_cfg[shot].get("cover", 0.78))
        spec = {"cover": cover, "span": "full", "head_top_frac": 0.25}

    cover = float(spec.get("cover", 0.78))
    span = spec.get("span", "full")
    head_frac = float(spec.get("head_top_frac", 0.25))

    y_top = head_y if np.isfinite(head_y) else top + 0.04 * bh
    if span == "chest":                       # 特写：胸以上
        y_ref = sh_y if np.isfinite(sh_y) else top + 0.20 * bh
        y_bot = y_ref + 0.5 * abs(y_ref - y_top)
        cx_t = sh_x if np.isfinite(sh_x) else bx
    elif span == "upper":                     # 中景：髋以上
        y_bot = hip_y if np.isfinite(hip_y) else top + 0.55 * bh
        cx_t = sh_x if np.isfinite(sh_x) else bx
    else:                                     # full：全身（wide / extreme_wide）
        y_bot = by + bh / 2.0
        cx_t = bx

    content_h = max(1.0, y_bot - y_top)
    z_t = cover * base_h / content_h
    view_h = base_h / max(z_t, 1e-6)
    # 头顶落在窗内 head_frac 处 => 头顶留白 = head_frac（你的"1/4、1/3"规格）
    cy_t = y_top + (0.5 - head_frac) * view_h
    return z_t, cx_t, cy_t
