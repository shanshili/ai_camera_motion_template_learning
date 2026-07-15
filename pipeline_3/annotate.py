"""
时间轴标注 + 字幕（preview 与 render 共用）

要标清楚三件事：
  1) 换景别的**时刻** —— 画在时间轴上（竖线 + 三角标）
  2) 景别**种类**     —— 字幕
  3) 该景别内的**运镜动作** —— 字幕

之前 preview 和 render 各画各的时间轴，标注含混。这里统一。
"""
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# 景别配色（近→远：暖→冷），字幕与时间轴共用同一套，便于对照
SHOT_COLOR = {
    "closeup":      (90, 90, 250),     # BGR 红
    "medium":       (70, 180, 250),    # 橙
    "wide":         (120, 220, 120),   # 绿
    "extreme_wide": (230, 180, 90),    # 蓝
}
SHOT_CN = {"closeup": "CLOSEUP", "medium": "MEDIUM",
           "wide": "WIDE", "extreme_wide": "EXTREME WIDE"}
MOVE_CN = {"follow": "FOLLOW", "push_in": "PUSH IN", "pull_out": "PULL OUT",
           "roll": "ROLL", "recenter": "RECENTER", "static": "STATIC",
           "whip": "WHIP", "orbit": "ORBIT"}


def seg_at(plan, i):
    """当前帧所在的分镜段。"""
    for s in (plan or []):
        if s["start_f"] <= i < s["end_f"]:
            return s
    return None


def achieved_shot(person, M, out_w, out_h):
    """
    ★从**成片画面**反量实际景别 —— 与 shot_plan 里的"设计意图"区分开。

    字幕原本只读 shot_plan 的 s["shot"]，那是 shotplan 阶段**想要**的景别。
    但 camera 层会被 max_zoom（画质预算）和安全框夹住：
    实测预览有 28% 的帧设计要 zoom=4.1、实际只做到 1.45 ——
    这些帧字幕写着 MEDIUM，画面里其实是远景。**标签在骗人。**

    做法：把 C位 关键点用同一个仿射矩阵 M 变换到成片坐标系，
    量「整个人（含画面外部分，按解剖比例补全）」占成片高的比例，再按同一套阈值定档。
    返回 (shot, cover) 或 (None, None)。
    """
    if not person or M is None:
        return None, None
    kp = {}
    for k in (person.get("keypoints") or []):
        xy = k.get("xy")
        if xy and len(xy) >= 2 and (k.get("confidence") is None or k["confidence"] >= 0.3):
            v = M @ np.array([float(xy[0]), float(xy[1]), 1.0])
            kp[k.get("name")] = (float(v[0]), float(v[1]))

    def mid(a, b):
        p, q = kp.get(a), kp.get(b)
        if p and q:
            return (p[1] + q[1]) / 2.0
        return (p or q or (None, None))[1]

    sh_y, hip_y = mid("left_shoulder", "right_shoulder"), mid("left_hip", "right_hip")
    nose_y = (kp.get("nose") or (None, None))[1]
    if sh_y is None or hip_y is None or hip_y <= sh_y:
        return None, None
    torso = hip_y - sh_y
    crown = (nose_y - 0.65 * (sh_y - nose_y)) if (nose_y is not None and sh_y > nose_y) \
        else sh_y - 0.5 * torso
    expect_ank = hip_y + 1.9 * torso
    cover = (expect_ank - crown) / max(1.0, out_h)     # 整个人占成片高的比例

    ank_y = mid("left_ankle", "right_ankle")
    knee_y = mid("left_knee", "right_knee")
    inside = lambda y: y is not None and -8 < y < out_h - 8
    if inside(ank_y):
        return ("extreme_wide" if cover < 0.45 else "wide"), cover
    if inside(knee_y):
        return ("medium" if cover > 1.35 else "wide"), cover
    if inside(hip_y):
        return "medium", cover
    return "closeup", cover
    """当前帧所在的分镜段。"""
    for s in (plan or []):
        if s["start_f"] <= i < s["end_f"]:
            return s
    return None


def shot_cut_frames(plan):
    """换景别的时刻（景别发生变化的段首帧）。运镜变化不算换景别。"""
    cuts = []
    prev = None
    for s in (plan or []):
        if prev is not None and s["shot"] != prev:
            cuts.append(int(s["start_f"]))
        prev = s["shot"]
    return cuts


def draw_timeline(w, h, plan, events, beats, n, cur_i, fps):
    """
    时间轴条：景别色带 + 换景别时刻(竖线+三角) + 拍点 + 事件 + 播放头。
    返回 h×w×3 的 BGR 图。
    """
    img = np.full((h, w, 3), 22, dtype=np.uint8)
    if n <= 0:
        return img
    x_of = lambda f: int(np.clip(f / max(1, n - 1) * (w - 1), 0, w - 1))

    band_y0, band_y1 = 26, h - 34
    # --- 景别色带 ---
    for s in (plan or []):
        c = SHOT_COLOR.get(s["shot"], (150, 150, 150))
        cv2.rectangle(img, (x_of(s["start_f"]), band_y0), (x_of(s["end_f"]), band_y1), c, -1)

    # --- 拍点（底部细刻度）---
    for b in (beats or []):
        if 0 <= b < n:
            x = x_of(b)
            cv2.line(img, (x, h - 30), (x, h - 24), (70, 70, 70), 1)

    # --- 事件（顶部小点）---
    for e in (events or []):
        f = int(e.get("frame", -1))
        if 0 <= f < n:
            cv2.circle(img, (x_of(f), 14), 2, (0, 215, 255), -1)

    # --- ★换景别时刻：白竖线 + 顶部三角标（最需要看清的东西）---
    for f in shot_cut_frames(plan):
        x = x_of(f)
        cv2.line(img, (x, band_y0 - 6), (x, band_y1 + 6), (255, 255, 255), 1)
        tri = np.array([[x, band_y0 - 7], [x - 4, band_y0 - 14], [x + 4, band_y0 - 14]], np.int32)
        cv2.fillPoly(img, [tri], (255, 255, 255))

    # --- 播放头 ---
    x = x_of(cur_i)
    cv2.line(img, (x, 4), (x, h - 4), (0, 0, 255), 2)

    # --- 时间刻度 ---
    for sec in range(0, int(n / max(1e-6, fps)) + 1, 5):
        f = int(sec * fps)
        if f < n:
            cv2.putText(img, f"{sec}s", (x_of(f) + 2, h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)
    return img


def draw_subtitle(frame, plan, i, fps, zoom=None, extra=None, actual=None):
    """
    左上角字幕，统一格式：  景别 / 运动    zoom=x.xx    已用/总时长

    ★注意 cv2.putText 只能画 ASCII，中文会变成 '???'（上一版图例就是这么糊掉的）。
      这里全部用英文标签。
    返回字幕框的矩形 (x0,y0,x1,y1)，供调用方避让、防止和人物标签重叠。
    """
    s = seg_at(plan, i)
    if s is None:
        return None
    shot, move = s["shot"], s.get("move", "")
    col = SHOT_COLOR.get(shot, (200, 200, 200))
    el = (i - s["start_f"]) / max(1e-6, fps)
    dur = (s["end_f"] - s["start_f"]) / max(1e-6, fps)

    l1 = f"{SHOT_CN.get(shot, shot.upper())} / {MOVE_CN.get(move, str(move).upper())}"
    l2 = (f"zoom={zoom:.2f}   " if zoom is not None else "") + f"{el:.1f}/{dur:.1f}s"
    lines = [(l1, 0.58, col), (l2, 0.44, (215, 215, 215))]
    # ★实际拍成的景别（从成片反量）。与设计不符时用红字标出 —— 这是最该看见的信息：
    #   camera 层被 max_zoom / 安全框夹住时，设计要中景、画面其实是远景。
    if actual is not None:
        a_shot, a_cover = actual
        if a_shot:
            same = (a_shot == shot)
            lines.append((f"actual: {SHOT_CN.get(a_shot, a_shot.upper())} (cover={a_cover:.2f})",
                          0.40, (170, 170, 170) if same else (80, 80, 255)))
    if extra:
        lines.append((extra, 0.38, (160, 160, 160)))

    pad, x0, y0 = 8, 12, 12
    wmax = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0] for t, sc, _ in lines)
    hsum = 20 * len(lines)
    x1, y1 = x0 + wmax + pad * 2 + 6, y0 + hsum + pad * 2
    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + 4, y1), col, -1)          # 左侧色条 = 景别色

    y = y0 + pad + 14
    for t, sc, c in lines:
        cv2.putText(frame, t, (x0 + pad + 6, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, 1, cv2.LINE_AA)
        y += 20

    if i in set(shot_cut_frames(plan)):
        cv2.putText(frame, "CUT", (x1 + 10, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return (x0, y0, x1, y1)


SKEL_EDGES = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
              ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
              ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
              ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
              ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
              ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
              ("nose", "left_shoulder"), ("nose", "right_shoulder")]


def _put_label(frame, text, pt, color, avoid=None, scale=0.55, thick=2):
    """
    画标签，并避开 avoid 矩形（字幕框）。重叠就把标签挪到框的下方。
    ★之前 C/B1/B2 直接画在人物 bbox 顶上，人一走到左上角就和字幕糊在一起。
    """
    x, y = int(pt[0]), int(max(14, pt[1]))
    if avoid:
        ax0, ay0, ax1, ay1 = avoid
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
        if not (x + tw < ax0 or x > ax1 or y < ay0 or y - th > ay1):
            y = ay1 + th + 6            # 压到字幕框上了 → 挪到框下面
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_subject_debug(frame, primary_person, backups, M=None, conf_th=0.3, avoid=None):
    """
    调试标注：C位**骨架**（粗，绿）+ 预C位（细框，黄，标 B1/B2）。
    M 为源→成片的仿射矩阵；坐标必须走同一个 M，否则必然与画面错位。
    avoid：字幕框矩形，标签会避开它。
    """
    def tx(pt):
        if M is None:
            return int(pt[0]), int(pt[1])
        v = M @ np.array([float(pt[0]), float(pt[1]), 1.0])
        return int(v[0]), int(v[1])

    for bi, b in enumerate(backups or []):
        box = b.get("box_xyxy")
        if not box:
            continue
        p0, p1 = tx((box[0], box[1])), tx((box[2], box[3]))
        cv2.rectangle(frame, p0, p1, (0, 215, 255), 1)
        _put_label(frame, f"B{bi+1}", (p0[0], p0[1] - 4), (0, 215, 255), avoid, 0.45, 1)

    if not primary_person:
        return frame
    kp = {}
    for k in (primary_person.get("keypoints") or []):
        xy = k.get("xy")
        if xy and len(xy) >= 2 and (k.get("confidence") is None or k["confidence"] >= conf_th):
            kp[k.get("name")] = xy
    for a, b in SKEL_EDGES:
        if a in kp and b in kp:
            cv2.line(frame, tx(kp[a]), tx(kp[b]), (80, 255, 80), 2, cv2.LINE_AA)
    for name, xy in kp.items():
        cv2.circle(frame, tx(xy), 3, (60, 220, 60), -1)
    box = primary_person.get("box_xyxy")
    if box:
        p0 = tx((box[0], box[1]))
        _put_label(frame, "C", (p0[0], p0[1] - 6), (80, 255, 80), avoid, 0.6, 2)
    return frame
