"""
骨架绘制
========
沿用 render_skeleton_only.py 的骨架边表与调色板。
提供两种绘制：
  - draw_person：在原坐标系画（骨架标记原始视频）
  - draw_person_transformed：把关键点过相机矩阵 M 再画（骨架标记运镜视频）
"""

import cv2
import numpy as np

from .transform import apply_to_points

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
]

EDGE_COLOR = (0, 255, 180)
POINT_COLOR = (30, 120, 255)
PERSON_PALETTE = [
    ((0, 255, 180), (30, 120, 255)),
    ((255, 210, 60), (255, 90, 40)),
    ((190, 120, 255), (120, 70, 255)),
    ((80, 220, 255), (40, 160, 255)),
    ((120, 255, 120), (70, 200, 70)),
]


def _points_dict(person, min_conf):
    """person -> {name: (x, y)}（过置信度）。"""
    pts = {}
    for kp in person.get("keypoints") or []:
        conf = kp.get("confidence")
        if conf is not None and conf < min_conf:
            continue
        xy = kp.get("xy")
        if not xy or len(xy) < 2:
            continue
        pts[kp["name"]] = (float(xy[0]), float(xy[1]))
    return pts


def _draw(frame, pts_int, line_width, point_radius, edge_color, point_color):
    for a, b in SKELETON_EDGES:
        if a in pts_int and b in pts_int:
            cv2.line(frame, pts_int[a], pts_int[b], edge_color, line_width, cv2.LINE_AA)
    for p in pts_int.values():
        cv2.circle(frame, p, point_radius, point_color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, point_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)


def draw_person(frame, person, min_conf=0.2, line_width=4, point_radius=5,
                edge_color=EDGE_COLOR, point_color=POINT_COLOR):
    if not person:
        return
    pts = _points_dict(person, min_conf)
    pts_int = {k: (int(round(v[0])), int(round(v[1]))) for k, v in pts.items()}
    _draw(frame, pts_int, line_width, point_radius, edge_color, point_color)


def draw_person_transformed(frame, person, M, min_conf=0.2, line_width=4,
                            point_radius=5, edge_color=EDGE_COLOR,
                            point_color=POINT_COLOR):
    """把关键点过 M（源->成片）后画到运镜画面上，保证与画面对齐。"""
    if not person:
        return
    pts = _points_dict(person, min_conf)
    if not pts:
        return
    names = list(pts.keys())
    src = np.array([pts[n] for n in names], dtype=np.float64)
    dst = apply_to_points(M, src)
    pts_int = {n: (int(round(dst[i, 0])), int(round(dst[i, 1])))
               for i, n in enumerate(names)}
    _draw(frame, pts_int, line_width, point_radius, edge_color, point_color)
