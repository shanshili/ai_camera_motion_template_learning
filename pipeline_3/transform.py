"""
统一相机变换 T
===============
把一帧 CameraParams(cx, cy, zoom, rot) 解释成"源画面 -> 成片画面"的仿射变换。
- 渲染最终画面：对帧做 warpAffine
- 骨架标记运镜视频：把关键点坐标过同一个矩阵，保证骨架与画面严格对齐

这样"运镜画面"和"运镜画面上的骨架"共用一套几何，绝不错位。
"""

import math

import cv2
import numpy as np


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def largest_source_view(src_w, src_h, out_w, out_h):
    """z=1 时源画面里能容纳输出比例的最大裁剪窗（宽,高）。"""
    src_ar = src_w / src_h
    out_ar = out_w / out_h
    if src_ar >= out_ar:
        view_h = float(src_h)
        view_w = view_h * out_ar
    else:
        view_w = float(src_w)
        view_h = view_w / out_ar
    return view_w, view_h


def effective_max_zoom(cfg, src_w, src_h, out_w, out_h):
    """考虑画质预算与 allow_upscale_for_demo 的最终最大 zoom。"""
    base_w, base_h = largest_source_view(src_w, src_h, out_w, out_h)
    quality_zmax = min(base_w / out_w, base_h / out_h)
    cam = cfg.get("camera", {})
    configured_max = float(cam.get("max_zoom", 1.45))
    if bool(cam.get("allow_upscale_for_demo", False)):
        return max(1.0, configured_max)
    return max(1.0, min(configured_max, quality_zmax))


def crop_rect(src_w, src_h, out_w, out_h, cx, cy, zoom):
    """裁剪窗（浮点）：中心 (cx,cy)、缩放 zoom 下的合法裁剪矩形。"""
    base_w, base_h = largest_source_view(src_w, src_h, out_w, out_h)
    z = max(1.0, float(zoom))
    view_w = max(4.0, base_w / z)
    view_h = max(4.0, base_h / z)
    cx = src_w / 2.0 if cx is None else float(cx)
    cy = src_h / 2.0 if cy is None else float(cy)
    x0 = clamp(cx - view_w / 2.0, 0.0, src_w - view_w)
    y0 = clamp(cy - view_h / 2.0, 0.0, src_h - view_h)
    return x0, y0, view_w, view_h, z


def camera_matrix(cam, src_w, src_h, out_w, out_h, max_zoom):
    """
    构造 2x3 仿射矩阵 M（源坐标 -> 成片坐标）：
      1) 平移裁剪中心到原点
      2) 旋转 -rot（图像坐标系 y 向下，正 rot 视觉逆时针）
      3) 缩放到输出尺寸
      4) 平移到成片中心
    """
    z = clamp(float(getattr(cam, "zoom", 1.0)), 1.0, float(max_zoom))
    x0, y0, view_w, view_h, z = crop_rect(src_w, src_h, out_w, out_h,
                                          getattr(cam, "cx", None),
                                          getattr(cam, "cy", None), z)
    rc_x = x0 + view_w / 2.0
    rc_y = y0 + view_h / 2.0
    sx = out_w / view_w
    sy = out_h / view_h
    theta = math.radians(float(getattr(cam, "rot", 0.0)))
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # 先绕裁剪中心旋转（源坐标系），再缩放，再平移到输出中心
    # p_out = S * R * (p_src - rc) + out_center
    a = sx * cos_t
    b = sx * sin_t
    c = -sy * sin_t
    d = sy * cos_t
    tx = out_w / 2.0 - (a * rc_x + b * rc_y)
    ty = out_h / 2.0 - (c * rc_x + d * rc_y)
    return np.array([[a, b, tx], [c, d, ty]], dtype=np.float64), (x0, y0, view_w, view_h, z)


def apply_to_frame(frame, M, out_w, out_h, interp=None):
    if interp is None:
        interp = cv2.INTER_CUBIC
    return cv2.warpAffine(frame, M, (out_w, out_h),
                          flags=interp, borderMode=cv2.BORDER_REPLICATE)


def apply_to_points(M, pts):
    """pts: (N,2) 源坐标 -> (N,2) 成片坐标。"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts
    ones = np.ones((pts.shape[0], 1))
    homo = np.hstack([pts, ones])
    return (homo @ M.T)
