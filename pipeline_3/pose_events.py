"""
姿态事件机（WP3 算法）
======================
沿用原 pose.py 的特征/事件检测，与具体检测器编号无关——只依赖"按关节名取的特征序列"。
现在检测器是 YOLO26-pose(COCO-17)，索引与 COCO17_MAP 一致，直接复用。

输入：每帧一个 (K,3) 的 [x,y,conf] 数组（无人则 None）。
输出：pose_frames（逐帧 bbox/质心/运动能量）+ pose_events（jump/spin/extension/freeze/level_change）。
"""

import numpy as np

# COCO-17 关节名 -> 索引（YOLO26-pose 顺序）
COCO17_MAP = {"nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
              "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
              "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
              "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16}


# ----------------------------- One Euro -----------------------------
class OneEuro:
    """One Euro 低延迟平滑：慢动小抖动强滤、快动少延迟。"""
    def __init__(self, fps, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.fps = fps; self.min_cutoff = min_cutoff
        self.beta = beta; self.d_cutoff = d_cutoff
        self.x_prev = None; self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, fps):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / fps
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x; return x
        dx = (x - self.x_prev) * self.fps
        ad = self._alpha(self.d_cutoff, self.fps)
        dxh = ad * dx + (1 - ad) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dxh)
        a = self._alpha(cutoff, self.fps)
        xh = a * x + (1 - a) * self.x_prev
        self.x_prev = xh; self.dx_prev = dxh
        return xh

    def reset(self, x=None):
        """
        硬切：清掉滤波状态，下一帧直接吐目标值、不做平滑过渡。
        裁剪相机没有真多机位，"切"就是让裁剪窗瞬间跳到新构图——
        不 reset 的话 One Euro + 限速会把切镜抹成一段缓慢推移。
        """
        self.x_prev = x
        self.dx_prev = 0.0


# ----------------------------- 特征 -----------------------------
def _named(kpts, jmap, conf_th=0.3):
    out = {}
    for name, idx in jmap.items():
        if idx < len(kpts) and kpts[idx][2] >= conf_th:
            out[name] = np.asarray(kpts[idx][:2], dtype=float)
        else:
            out[name] = None
    return out


def features_from_kpts(kpts, jmap, conf_th=0.3):
    """从一帧关键点算特征：质心、身高、肩/胯宽、肢体伸展度、主体框。"""
    kpts = np.asarray(kpts, dtype=float)
    j = _named(kpts, jmap, conf_th)
    valid = np.array([kpts[i][:2] for i in jmap.values()
                      if i < len(kpts) and kpts[i][2] >= conf_th])
    if len(valid) < 3:
        return None
    centroid = valid.mean(axis=0)
    body_h = max(1.0, valid[:, 1].max() - valid[:, 1].min())

    def width(a, b):
        if j[a] is not None and j[b] is not None:
            return abs(j[a][0] - j[b][0])
        return np.nan

    shoulder_w = width("l_shoulder", "r_shoulder")
    hip_w = width("l_hip", "r_hip")

    limbs = [j[n] for n in ("l_wrist", "r_wrist", "l_ankle", "r_ankle")
             if j[n] is not None]
    spread = (np.mean([np.linalg.norm(p - centroid) for p in limbs]) / body_h
              if limbs else np.nan)

    x0, y0 = valid.min(axis=0); x1, y1 = valid.max(axis=0)
    bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
    return {"centroid": centroid, "body_h": body_h, "shoulder_w": shoulder_w,
            "hip_w": hip_w, "spread": spread, "bbox": bbox, "named": j}


def features_from_bbox(bbox):
    """从主体框出特征（无关键点时的退化路径，肩宽/伸展度为 NaN）。"""
    if bbox is None:
        return None
    x, y, w, h = bbox
    return {"centroid": np.array([x + w / 2.0, y + h / 2.0]),
            "body_h": max(1.0, float(h)), "shoulder_w": np.nan,
            "hip_w": np.nan, "spread": np.nan,
            "bbox": [float(x), float(y), float(w), float(h)], "named": {}}


# ----------------------------- 事件检测 -----------------------------
def _rolling_median(x, w):
    n = len(x); out = np.empty(n); h = w // 2
    for i in range(n):
        seg = x[max(0, i - h):min(n, i + h + 1)]
        seg = seg[np.isfinite(seg)]           # 手动剔 NaN，避免 all-NaN 切片告警
        out[i] = np.median(seg) if seg.size else np.nan
    return out


def _runs(mask, min_len):
    runs = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def detect_events(feat_series, fps):
    """在特征序列上检测姿态事件，返回 (events, motion)。"""
    from scipy.signal import find_peaks
    n = len(feat_series)
    if n == 0:
        return [], np.zeros(0)
    cy = np.array([f["centroid"][1] if f else np.nan for f in feat_series])
    body_h = np.nanmedian([f["body_h"] for f in feat_series if f]) or 1.0
    sw = np.array([f["shoulder_w"] if f else np.nan for f in feat_series])
    spread = np.array([f["spread"] if f else np.nan for f in feat_series])
    cx = np.array([f["centroid"][0] if f else np.nan for f in feat_series])

    vel = np.full(n, np.nan)
    for i in range(1, n):
        if not (np.isnan(cx[i]) or np.isnan(cx[i - 1])):
            vel[i] = np.hypot(cx[i] - cx[i - 1], cy[i] - cy[i - 1]) / body_h * fps
    motion = np.nan_to_num(vel)

    events = []
    min_dist = max(1, int(0.2 * fps))

    # 起跳：质心高于基线（图像 y 向下，跳起 = y 减小）
    base_cy = _rolling_median(cy, int(1.0 * fps) | 1)
    above = np.nan_to_num((base_cy - cy) / body_h)
    pk, props = find_peaks(above, height=0.12, distance=min_dist)
    for k, p in enumerate(pk):
        events.append({"frame": int(p), "type": "jump",
                       "intensity": round(float(props["peak_heights"][k]), 3)})

    # 伸展：肢体伸展度显著峰
    med_sp = np.nanmedian(spread)
    std_sp = np.nanstd(spread)
    sp = np.nan_to_num(spread, nan=med_sp if np.isfinite(med_sp) else 0.0)
    if np.isfinite(med_sp):
        pk, _ = find_peaks(sp, prominence=max(0.06, 0.4 * (std_sp or 0.0)), distance=min_dist)
        for p in pk:
            if sp[p] > med_sp * 1.15:
                events.append({"frame": int(p), "type": "extension",
                               "intensity": round(float(sp[p]), 3)})

    # 旋转：肩投影宽持续显著收窄
    med_sw = np.nanmedian(sw)
    if np.isfinite(med_sw) and med_sw > 1e-6:
        narrow = np.nan_to_num(sw, nan=med_sw) < 0.55 * med_sw
        for s, e in _runs(narrow, min_len=max(2, int(0.12 * fps))):
            events.append({"frame": int((s + e) // 2), "type": "spin",
                           "intensity": round(float(1 - np.nanmin(sw[s:e + 1]) / (med_sw + 1e-9)), 3)})

    # 定格：运动能量持续很低
    still = motion < 0.05
    for s, e in _runs(still, min_len=max(2, int(0.3 * fps))):
        events.append({"frame": int(s), "type": "freeze",
                       "intensity": round(float(e - s) / fps, 3)})

    # 重心变化：长窗基线显著平移
    win = int(0.4 * fps) | 1
    base = _rolling_median(cy, win)
    for i in range(win, n - win):
        d = (np.nanmedian(base[i:i + win]) - np.nanmedian(base[i - win:i])) / body_h
        if abs(d) > 0.25:
            events.append({"frame": int(i), "type": "level_change",
                           "intensity": round(float(abs(d)), 3)})
            break

    events.sort(key=lambda e: e["frame"])
    return events, motion


# ----------------------------- 收尾/打包 -----------------------------
def _finalize(feats, fps):
    fx, fy = OneEuro(fps), OneEuro(fps)
    for f in feats:
        if f is not None:
            f["centroid"] = np.array([fx(f["centroid"][0]), fy(f["centroid"][1])])
    events, motion = detect_events(feats, fps)
    pose_frames = []
    for i, f in enumerate(feats):
        if f is None:
            pose_frames.append({"frame": i, "bbox": None, "centroid": None,
                                "motion_energy": 0.0})
        else:
            pose_frames.append({
                "frame": i,
                "bbox": [round(v, 1) for v in f["bbox"]],
                "centroid": [round(float(f["centroid"][0]), 1),
                             round(float(f["centroid"][1]), 1)],
                "motion_energy": round(float(motion[i]), 4),
            })
    return {"fps": float(fps), "n_frames": len(feats),
            "pose_frames": pose_frames, "pose_events": events}


def analyze_pose_from_kpts(kpts_series, fps, jmap=COCO17_MAP, conf_th=0.3):
    """从关键点序列出特征/事件。可脱离图像直接调用（便于测试）。"""
    feats = [features_from_kpts(k, jmap, conf_th) if k is not None else None
             for k in kpts_series]
    return _finalize(feats, float(fps))
