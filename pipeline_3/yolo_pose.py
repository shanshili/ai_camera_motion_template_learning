"""
姿态纯助手（无 YOLO / 无 ffmpeg 依赖）
=====================================
多人检测+跟踪已移到独立包 multi_person_tracking（run_tracking.py，需 ultralytics）。
这里只保留 subject.py / analysis.py 复用的纯函数：
  - 主体几何：_person_center / _person_area / _person_diag
  - 短间隙插值：fill_primary_gaps（+ _interp_person）
  - 打包：primary_series_to_kpts / build_primary_records
records schema 与 multi_person_tracking 产出的 tracked_keypoints.json 一致：
  {"frame_index","original_shape","people":[
     {"person_index","tracker_id","box_xyxy","score",
      "keypoints":[{"name","xy","xyn","confidence"}]}]}
"""

import numpy as np

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ----------------------------- 主体几何 -----------------------------
def _person_center(person):
    box = person.get("box_xyxy")
    if box and len(box) >= 4:
        return ((float(box[0]) + float(box[2])) / 2.0,
                (float(box[1]) + float(box[3])) / 2.0)
    pts = [kp["xy"] for kp in (person.get("keypoints") or [])
           if kp.get("xy") and len(kp["xy"]) >= 2]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _person_area(person):
    box = person.get("box_xyxy")
    if box and len(box) >= 4:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return 0.0


def _person_diag(person):
    box = person.get("box_xyxy")
    if box and len(box) >= 4:
        w = max(0.0, box[2] - box[0]); h = max(0.0, box[3] - box[1])
        return float(np.hypot(w, h))
    return 0.0


# ----------------------------- 短间隙插值 -----------------------------
def _interp_person(p_before, p_after, t, frame_index):
    """两帧主体 person 之间按 t∈(0,1) 线性插值，生成补帧 person。"""
    kb = {kp.get("name"): kp for kp in (p_before.get("keypoints") or [])}
    ka = {kp.get("name"): kp for kp in (p_after.get("keypoints") or [])}
    kps = []
    for name in KEYPOINT_NAMES:
        a = kb.get(name); b = ka.get(name)
        if a and b and a.get("xy") and b.get("xy") \
                and len(a["xy"]) >= 2 and len(b["xy"]) >= 2:
            x = a["xy"][0] * (1 - t) + b["xy"][0] * t
            y = a["xy"][1] * (1 - t) + b["xy"][1] * t
            ca = a.get("confidence"); cb = b.get("confidence")
            ca = 1.0 if ca is None else float(ca)
            cb = 1.0 if cb is None else float(cb)
            conf = (ca * (1 - t) + cb * t) * 0.9
            kps.append({"name": name, "xy": [x, y], "xyn": None, "confidence": conf})
        else:
            src = a if (a and a.get("xy")) else b
            if src and src.get("xy"):
                kps.append({"name": name, "xy": list(src["xy"]),
                            "xyn": None, "confidence": (src.get("confidence") or 0.5) * 0.8})
    bb = p_before.get("box_xyxy"); ba = p_after.get("box_xyxy")
    box = None
    if bb and ba and len(bb) >= 4 and len(ba) >= 4:
        box = [bb[i] * (1 - t) + ba[i] * t for i in range(4)]
    elif bb:
        box = list(bb)
    elif ba:
        box = list(ba)
    return {"person_index": 0, "tracker_id": None, "box_xyxy": box, "score": 0.5,
            "keypoints": kps, "_interpolated": True}


def fill_primary_gaps(primary, max_gap=15):
    """对主体逐帧序列里长度 <= max_gap 的 None 空洞两端线性插值补齐。"""
    primary = list(primary)
    n = len(primary)
    i = 0
    filled = 0
    while i < n:
        if primary[i] is not None:
            i += 1
            continue
        j = i
        while j < n and primary[j] is None:
            j += 1
        left = primary[i - 1] if i - 1 >= 0 else None
        right = primary[j] if j < n else None
        gap = j - i
        if left is not None and right is not None and gap <= max_gap:
            for k in range(i, j):
                t = (k - i + 1) / (gap + 1)
                primary[k] = _interp_person(left, right, t, k)
                filled += 1
        i = j
    if filled:
        print(f"[pose] 主体短间隙插值补齐 {filled} 帧（阈值 {max_gap}）")
    return primary


# ----------------------------- 打包 -----------------------------
def primary_series_to_kpts(primary_per_frame, min_conf=0.0):
    """主体逐帧 person → (17,3) 的 [x,y,conf] 序列（COCO17 顺序），供事件机使用。"""
    series = []
    for person in primary_per_frame:
        if not person:
            series.append(None); continue
        arr = np.zeros((17, 3), dtype=float)
        by_name = {kp.get("name"): kp for kp in (person.get("keypoints") or [])}
        for i, name in enumerate(KEYPOINT_NAMES):
            kp = by_name.get(name)
            if kp and kp.get("xy") and len(kp["xy"]) >= 2:
                conf = kp.get("confidence")
                conf = 1.0 if conf is None else float(conf)
                arr[i] = [float(kp["xy"][0]), float(kp["xy"][1]), conf]
        series.append(arr)
    return series


def build_primary_records(records, primary_per_frame):
    """生成 primary_records：每帧带 primary_person，供 camera 几何 / 渲染骨架复用。"""
    out = []
    for rec, person in zip(records, primary_per_frame):
        out.append({
            "frame_index": rec.get("frame_index"),
            "original_shape": rec.get("original_shape"),
            "primary_person": person,
            "people": rec.get("people") or [],
        })
    return out
