"""
主体选择层 · 多人 → C位角色化（方案 §4）
==========================================
替换旧的 yolo_pose.select_primary_track（单主体锁定）。

输入：多人逐帧 records（带 tracker_id，来自 ByteTrack 前端 / tracked_keypoints.json）。
输出：primary_per_frame（每帧 C位 person，schema 与旧 primary_person 完全一致，
      直接喂 build_primary_records / primary_series_to_kpts / camera._subject_geometry）
      + meta（逐帧 compose_mode / group_box / 焦点切换帧）。

设计要点（对应方案条目）：
  §4.1 C位分数：居中度 P + 相对尺度 Z + 相对运动 M + 正面度 F (+ 事件加成 E)
  §4.2 迟滞状态机：焦点绑“上一帧焦点位置的最近邻”，不绑 tracker_id；
        换人要“够强(delta) 且 够久(tau)”，否则现任续任（incumbent_bonus）
  §4.3 FEATURE / GROUP：注意力集中度判决，带双阈值迟滞，防逐帧翻转
  §4.4 对接：C位 person 走原单人机器；GROUP 的群体框留在 meta.group_box，
        由 camera 的 box 输入路径消费（本模块不改 camera）

★为什么焦点不绑 tracker_id：
  ByteTrack 在快速换位/交叉遮挡时会换 id，焦点若绑 id 就会跟着跳。
  这里焦点只认“离上一帧焦点位置最近的人”，对 id 抖动免疫。
  tracker_id 仅用于跨帧计算运动速度（M 项），不参与焦点身份。

★与 canonical frame 的关系（方案 §3，属阶段2，先于本层）：
  Z（相对尺度）、P（居中度）此处用“帧内相对量 / 源像素”近似。固定机位单源够用；
  一旦引入 canonical frame，把 _center_score / _scale_score 两个函数换成
  canonical 相对量即可，本模块其余逻辑不变。
"""

import math

import numpy as np

from . import yolo_pose as yp   # 复用其纯函数：_person_center/_person_area/fill_primary_gaps/build_primary_records
from .log import Progress


# ----------------------------- 默认配置 -----------------------------
DEFAULT_CFG = {
    # §4.1 各项权重（可作为模板的一部分，不同风格“看谁”偏好不同）
    #   ★居中绝对优先：C位就是站在队形中间的人。P 已按群体展布归一（满量程 0..1），
    #     权重 2.0 让它压过其它项之和 —— 边角的人不可能靠 尺度/正面度 翻盘。
    #   ★尺度降权：固定机位下前排的人天然更大，但"离镜头近" ≠ "是C位"。
    #   ★运动是条件项：只在「队伍大部分人动作较少」时才生效（见 motion_gate）。
    "weights": {"center": 3.0, "scale": 0.25, "motion": 2.0, "frontal": 0.15, "event": 0.0},
    # 运动闸门：群体运动中位数低于此 → 判「大家都不太动」→ 才让"动作最大的"夺位
    "motion_gate": {"group_still_th": 0.08, "center_damp_when_still": 0.2},
    # ★居中度参考点：frame=原始画面中心（推荐，最稳；固定机位下舞台中心≈画面中心）
    #   group=群体质心（依赖"这帧检测到谁"，漏检会让质心整体偏移 → C位跟着漂）
    "center_ref": "frame",
    # 画面中央区域：x_frac=0.333 即三等分的中间那一份。
    #   现任移出该区域 → 触发换 C；候选须在该区域内/正移向中央/离中央够近(near_frac)。
    # x_frac    : 接任资格区（紧）——五等分取中间，只有真的在正中才能接任 C位
    # near_frac : 资格放宽区——在此区内且正移向中央的人也有资格
    # exit_frac : 切走区（宽）——现任走出这里才触发换人。必须比 x_frac 宽得多，
    #             否则舞者正常走位就会不停触发换人（表现为"后段 C位 乱飞"）
    # exit_grace_sec: 走出后需持续多久才算数，防抖一下就换
    "center_zone": {"x_frac": 0.2, "near_frac": 0.4},
    # ID 断裂后同一个人在相邻帧的位移上限（按身高比例）。ByteTrack 在密集遮挡下
    #   ID 会疯狂断裂（11 人能churn出 220 条 track），必须按位置把它们续起来，
    #   否则 DP 会去挑"ID 活得久的人"而不是"在中央的人"。
    "reid_body_frac": 0.35,
    # ★段落锁定：一个唱段内 C位 不换人（即便他移动），只在段落边界/硬切附近才允许换。
    #   这是对“换唱段才换C位、切景才短暂给别人”这一实拍规律的直接编码，
    #   也是治「C位乱聚焦」最有效的一条——把换人机会从每帧降到每段一次。
    # ★段落锁定（全局DP的换人代价）：
    #   switch_cost λ = 段内换 C位 要付的一次性代价。它就是"迟滞"，
    #     但是全局且前瞻的：短暂抖动累积收益<λ→不换；持续变化累积收益>λ→换，
    #     且换在最优时刻，不是事后补救。调大=更黏，调小=更灵敏。
    #   switch_window_sec: 段落边界/硬切前后多久内换人免费（λ=0）。
    "section_lock": {"enabled": True, "switch_cost": 45.0,
                     "switch_cost_at_boundary": 5.0, "switch_window_sec": 0.5},
    # §4.3 FEATURE/GROUP：集中度阈值 + 双阈值迟滞带 + 模式切换需持续帧数
    "compose": {"feature_th": 0.35, "release_margin": 0.10, "hold_sec": 0.5},
    "fill_gap_frames": 15,   # C位短间隙插值（复用旧逻辑）
    "min_kp_conf": 0.3,
}


def _merge_cfg(cfg):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CFG.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


# ----------------------------- 几何小工具 -----------------------------
def _kp(person, name, conf_th):
    for kp in person.get("keypoints") or []:
        if kp.get("name") == name and kp.get("xy") and len(kp["xy"]) >= 2:
            c = kp.get("confidence")
            if c is None or c >= conf_th:
                return float(kp["xy"][0]), float(kp["xy"][1])
    return None


def _body_height(person, conf_th=0.3):
    """
    主体身高（像素）。★必须过滤低置信关键点：YOLO 对未检出的关节常给 (0,0) 占位，
    若把它算进来，min(ys)=0 会让身高被撑到整幅画面，
    进而让 M(运动/身高) 与 F(肩宽/身高) 全部失真。
    """
    ys = []
    for kp in (person.get("keypoints") or []):
        xy = kp.get("xy")
        if not xy or len(xy) < 2:
            continue
        c = kp.get("confidence")
        if c is not None and c < conf_th:
            continue
        ys.append(xy[1])
    if len(ys) >= 2:
        return max(1.0, max(ys) - min(ys))
    box = person.get("box_xyxy")
    if box and len(box) >= 4:
        return max(1.0, float(box[3]) - float(box[1]))
    return 1.0


def _union_box(people):
    xs0, ys0, xs1, ys1 = [], [], [], []
    for p in people:
        b = p.get("box_xyxy")
        if b and len(b) >= 4:
            xs0.append(float(b[0])); ys0.append(float(b[1]))
            xs1.append(float(b[2])); ys1.append(float(b[3]))
    if not xs0:
        return None
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


# ----------------------------- §4.1 各分项 -----------------------------
def _center_ref_and_norm(people, frame_w, frame_h, cfg):
    """
    居中度的参考点与归一尺度。
    ★默认用「原始画面中心」而不是群体质心：
      群体质心依赖"这一帧检测到了哪些人"，漏检/遮挡会让质心整体偏移，
      C位判定就跟着漂。画面中心是固定机位下最稳的参考——舞台中心≈画面中心。
    """
    ref_mode = str(cfg.get("center_ref", "frame"))
    if ref_mode == "group":
        cs = [c for c in (yp._person_center(p) for p in people) if c]
        if cs:
            gc = (float(np.mean([c[0] for c in cs])), float(np.mean([c[1] for c in cs])))
            d = [math.dist(c, gc) for c in cs]
            return gc, max(max(d), 1e-6)
    return (frame_w / 2.0, frame_h / 2.0), frame_w / 2.0


def centers_of(p):
    return yp._person_center(p)


def _in_center_zone(c, frame_w, zone_frac):
    """是否在画面中央区域（默认三等分的中间那一份）。"""
    return abs(c[0] - frame_w / 2.0) <= (zone_frac * frame_w) / 2.0


def _moving_to_center(c, prev_c, frame_w):
    """是否正在向画面中央移动。"""
    if prev_c is None:
        return False
    return abs(c[0] - frame_w / 2.0) < abs(prev_c[0] - frame_w / 2.0) - 1.0


def _center_score(c, ref, norm):
    """
    居中度：离画面中心越近越高。
    ★只看横向。你要的是「画面横向中间的人」，纵向不该参与——
      站得靠后的人 y 更小，二维距离会把他判得更不居中，这不合理。
    ★归一用画面半宽：正中 P=1、最左/最右 P=0，满量程区分。
    """
    if c is None:
        return 0.0
    return float(np.clip(1.0 - abs(c[0] - ref[0]) / (norm + 1e-6), 0.0, 1.0))


def _scale_score(area, max_area):
    """相对尺度：近大远小。归一到帧内最大主体。"""
    return float(area / (max_area + 1e-6)) if max_area > 0 else 0.0


def _frontal_score(person, conf_th):
    """正面度：双肩投影宽相对身高越大越正面（侧身时肩宽收窄）。未知返回中性 0.5。"""
    ls = _kp(person, "left_shoulder", conf_th)
    rs = _kp(person, "right_shoulder", conf_th)
    if not (ls and rs):
        return 0.5
    return float(np.clip(abs(ls[0] - rs[0]) / (0.35 * _body_height(person) + 1e-6), 0.0, 1.0))


def _frame_scores(people, prev_center_by_tid, frame_w, frame_h, cfg):
    """算这一帧每个人的 C位分数（与 people 等长）。"""
    w = cfg["weights"]
    conf_th = cfg["min_kp_conf"]
    if not people:
        return []

    centers = [yp._person_center(p) for p in people]
    areas = [yp._person_area(p) for p in people]
    ref, norm = _center_ref_and_norm(people, frame_w, frame_h, cfg)
    max_area = max(areas) if areas else 1.0

    # 运动：需要同一 tracker_id 的上一帧中心；缺失记 NaN，用中位数近似“别人”的量级
    motions = []
    for p, c in zip(people, centers):
        tid = p.get("tracker_id")
        bh = _body_height(p, conf_th)
        if c and tid is not None and tid in prev_center_by_tid and bh > 1:
            motions.append(math.dist(c, prev_center_by_tid[tid]) / bh)
        else:
            motions.append(np.nan)
    finite = [m for m in motions if np.isfinite(m)]
    med_motion = float(np.median(finite)) if finite else 0.0
    # ★"没有运动数据" ≠ "运动为零"。片头/跟踪断裂时拿不到上一帧中心，
    #   若把它当成"大家都不动"，就会切到运动主导模式、把居中权重打折，
    #   于是靠身高噪声随便选个人——再叠加段落锁定，这个错误会被焊死一整段。
    #   必须至少半数人有真实运动数据，才允许判定"群体静止"。
    have_motion = len(finite) >= max(2, (len(people) + 1) // 2)

    # ★运动闸门：你的规则是「两种模式」，不是一个加权求和。
    #   模式1（队伍普遍在动）：居中主导，运动完全不参与——
    #        否则边角的人靠狂动累加就能抢走 C位（实测就是这么锁到边角去的）。
    #   模式2（队伍大部分人动作较少）：运动主导，居中降权——
    #        "别人静他动"的那个人就是此刻的焦点。
    still_th = float(cfg.get("motion_gate", {}).get("group_still_th", 0.08))
    damp = float(cfg.get("motion_gate", {}).get("center_damp_when_still", 0.4))
    group_still = have_motion and (med_motion < still_th)
    w_center = w["center"] * (damp if group_still else 1.0)
    w_motion = w["motion"] if group_still else 0.0

    scores = []
    for k, p in enumerate(people):
        P = _center_score(centers[k], ref, norm)
        Z = _scale_score(areas[k], max_area)
        m = motions[k]
        # M 与 P/Z/F 同量纲 [0,1]：饱和函数 r/(1+r)，r=1(与他人相当)→0.5
        if np.isfinite(m) and med_motion > 1e-6:
            r = m / med_motion
            M = float(r / (1.0 + r))
        else:
            M = 0.5
        F = _frontal_score(p, conf_th)
        E = 0.0
        scores.append(w_center * P + w["scale"] * Z + w_motion * M
                      + w["frontal"] * F + w["event"] * E)
    return scores


# ----------------------------- §4.2 迟滞焦点状态机 -----------------------------
# 注：FocusSelector（在线迟滞状态机）已被全局 DP 取代并删除。
#     incumbent_bonus / tau 迟滞 / 最近邻续任 / exit_grace 这些补丁，
#     全都是「只能看过去」逼出来的。既然是离线管线、未来已知，
#     就该整体求最优，而不是逐帧猜。


class ComposeState:
    def __init__(self, fps, cfg):
        c = cfg["compose"]
        self.th = float(c["feature_th"])
        self.rel = float(c["release_margin"])
        self.hold = max(1, int(round(float(c["hold_sec"]) * fps)))
        self.mode = "FEATURE"
        self.cnt = 0

    def update(self, scores):
        if len(scores) < 2:
            target = "FEATURE"
        else:
            s = sorted(scores, reverse=True)
            conc = (s[0] - s[1]) / (s[0] + 1e-6)   # 集中度：top1 相对 top2 领先多少
            # 双阈值迟滞：进入 GROUP 需跌破 th-rel，回到 FEATURE 需超过 th+rel
            if self.mode == "FEATURE":
                target = "FEATURE" if conc > (self.th - self.rel) else "GROUP"
            else:
                target = "FEATURE" if conc > (self.th + self.rel) else "GROUP"
        if target == self.mode:
            self.cnt = 0
        else:
            self.cnt += 1
            if self.cnt >= self.hold:
                self.mode, self.cnt = target, 0
        return self.mode


# ----------------------------- 主入口 -----------------------------
# ===================================================================
# ★C位选择：全局动态规划（离线管线，可以看全片，不该假装自己是实时摄像机）
#
# 之前是「因果在线状态机」：只看过去，用 incumbent_bonus / tau 迟滞 /
# exit_grace 去猜未来。那些补丁全都是「不知道未来」逼出来的，而且必然有代价：
#   · 「走开了才切」永远迟到 exit_grace 秒 —— 可未来几帧明摆着他要走；
#   · 段落边界靠单帧分数选人 —— 那一帧噪声大就选错，然后锁死一整段；
#   · 现任靠最近邻续任 → 身份随机游走。
#
# 但 tracked_keypoints.json 是全片一次性读进来的。既然未来已知，就该整体求最优：
#
#   C(t,k) = e(t,k) + max( C(t-1,k),                      # 留任
#                          max_{j≠k} C(t-1,j) − λ(t) )    # 换人，付一次代价
#
# 回溯 argmax 得到全片最优 C位 序列。λ(t) 在段落边界/硬切处=0（免费换），
# 段内=λ_mid。λ 本质就是迟滞，但它是全局且前瞻的：
#   · 短暂抖动：累积收益 < λ → 不换（自动抗噪，不需要 tau）
#   · 持续变化：累积收益 > λ → 换，且换在**最优时刻**，不是事后补救
# 这与 music.py 里的 Ellis 拍点跟踪 DP 同构。
# ===================================================================

def _build_frame_table(records, frame_w, frame_h, fps, cfg):
    """
    逐帧算每个人的 C位分数 → emit[n,M] / pos[n,M,2] / tid[n,M]。

    ★状态 = 「本帧的第几个人」(M = 全片单帧最多人数, ~15)，不是「全片哪条 track」。
      按 track 建状态会被 ID 断裂撑爆：11 个舞者能churn出 675 条 track，
      DP 的 K² 转移矩阵直接把耗时推到 43 秒。按帧内槽位建状态，M≈15，瞬间完成。
    """
    n = len(records)
    M = max(1, max((len(r.get("people") or []) for r in records), default=1))
    emit = np.full((n, M), -np.inf)
    pos = np.full((n, M, 2), np.nan)
    tid = np.full((n, M), -1, dtype=np.int64)
    people_at = [None] * n
    prev_center_by_tid = {}
    _prog = Progress(n, "C位打分", every_frac=0.25, min_step=500)
    for i, rec in enumerate(records):
        _prog.update(i)
        people = (rec.get("people") or [])[:M]
        people_at[i] = people
        scores = _frame_scores(people, prev_center_by_tid, frame_w, frame_h, cfg)
        for m, (p, sc) in enumerate(zip(people, scores)):
            emit[i, m] = sc
            c = yp._person_center(p)
            if c:
                pos[i, m] = c
            t = p.get("tracker_id")
            if t is not None:
                tid[i, m] = int(t)
        for p in people:
            t = p.get("tracker_id"); c = yp._person_center(p)
            if t is not None and c:
                prev_center_by_tid[t] = c
    _prog.done()
    return emit, pos, tid, people_at


def _free_transition(pos_prev, pos_cur, tid_prev, tid_cur, reid_px):
    """
    算「哪些转移是免费的」——即 j→m 属于「同一个人」而非「换人」。

    free[j,m] 为真当且仅当：
      · tid 相同                                  → 同一条 track 续任
      · 或 (旧 track 在本帧确实消失了) 且
           (m 是离 pos_prev[j] 最近的人, 距离 < reid_px) 且
           (次近的人比它远一倍以上 → 匹配无歧义)   → 同一个人换了工牌

    ★这里有两个曾经致命的坑：
      1) 不能用 np.fill_diagonal(Lam,0) 当「留任免费」。状态是「本帧的第 m 个人」，
         槽位序号由 YOLO 检测顺序决定，t-1 的第 3 人 ≠ t 的第 3 人。
         把「槽位号相同」当「同一个人」，DP 就能免费在不同人之间跳。
      2) re-ID 不能只看「距离 < 半径」。相邻舞者间距 ~140px、半径 114px，
         且旧 ID 明明还在场时也允许 re-ID —— 等于宣布「跳到隔壁人免费」。
      两坑叠加 → DP 退化成逐帧取 argmax → C位 在两边疯狂摇摆。
    """
    M = pos_cur.shape[0]
    same_id = (tid_prev[:, None] == tid_cur[None, :]) & (tid_prev[:, None] >= 0)
    j_alive = same_id.any(axis=1)                 # 旧 track 本帧是否还在
    d = np.linalg.norm(pos_cur[None, :, :] - pos_prev[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    # 每个 j 的最近/次近
    order = np.argsort(d, axis=1)
    nearest = order[:, 0]
    d1 = d[np.arange(M), nearest]
    d2 = d[np.arange(M), order[:, 1]] if M > 1 else np.full(M, np.inf)
    unambiguous = (d1 < reid_px) & (d2 > 2.0 * d1)   # 次近远一倍以上才算认得准
    reid_ok = np.zeros_like(same_id)
    rows = np.where((~j_alive) & unambiguous)[0]     # ★只对「确实消失了」的 j 开放
    reid_ok[rows, nearest[rows]] = True
    return same_id | reid_ok


def _viterbi_focus(emit, pos, tid, lam, reid_px):
    """
    C(t,m) = emit(t,m) + max_j ( C(t-1,j) − Λ(t,j,m) )
      Λ = 0        若 j→m 是「同一个人」（同 tid，或旧 track 消失后无歧义的位置续接）
        = lam[t]   否则（真换人，付一次代价）

    ★-inf 传染：某帧一个人都没检测到 → 全体 emit=-inf → 全体 C=-inf →
      之后加什么都是 -inf，DP 从此瘫痪、后面靠 argmax 乱选。故无人帧原样带过。

    O(n·M²)，M≈15 → 全片毫秒级。返回每帧最优槽位 + 免费转移矩阵（供统计真实换人）。
    """
    n, M = emit.shape
    NEG = -1e18
    e = np.where(np.isfinite(emit), emit, NEG)
    C = e[0].copy()
    bp = np.zeros((n, M), dtype=np.int32)
    bp[0] = np.arange(M)
    free_at = [None] * n
    idx = np.arange(M)
    for t in range(1, n):
        if not np.isfinite(emit[t]).any():        # 本帧无人：原样带过，别让 -inf 传染
            bp[t] = idx
            continue
        free = _free_transition(pos[t - 1], pos[t], tid[t - 1], tid[t], reid_px)
        free_at[t] = free
        Lam = np.where(free, 0.0, lam[t])         # ★不再有"槽位对角线免费"
        M_ = C[:, None] - Lam
        bp[t] = np.argmax(M_, axis=0)
        C = e[t] + M_[bp[t], idx]
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmax(C))
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[t][path[t]]
    return path, free_at


def select_subject(records, frame_w, frame_h, fps, cfg=None,
                   sections=None, cut_frames=None):
    """
    多人 records → 逐帧 C位 person（与旧 primary_person 同 schema）+ meta。

    ★全局最优：一次性看完全片再决定每一帧的 C位（离线管线本就该这样）。
    sections    : 音乐段落 [{"start_f","end_f","label"}] → 边界处换人免费
    cut_frames  : 硬切帧号 → 同上
    """
    cfg = _merge_cfg(cfg)
    n = len(records)
    if n == 0:
        return [], {"compose_mode": [], "group_box": [], "focus_switch_frames": []}

    emit, pos, tid, people_at = _build_frame_table(records, frame_w, frame_h, fps, cfg)
    K = emit.shape[1]

    # ---- λ(t)：换人代价。段落边界/硬切处免费，段内昂贵 ----
    lock = cfg.get("section_lock", {})
    lam_mid = float(lock.get("switch_cost", 45.0))
    lam_bnd = float(lock.get("switch_cost_at_boundary", 5.0))
    lam = np.full(n, lam_mid, dtype=np.float64)
    marks = []
    if bool(lock.get("enabled", True)):
        for s in (sections or []):
            marks.append(int(s.get("start_f", 0)))
        marks += [int(c) for c in (cut_frames or [])]
        win = max(1, int(round(float(lock.get("switch_window_sec", 0.5)) * fps)))
        for f in marks:
            # ★边界处换人「便宜」但不能「免费」。λ=0 会让 DP 在窗口内退化成
            #   逐帧取 argmax —— 噪声一抖就翻，实测 6 个边界能抖出 115 次换人。
            #   给一个小的正代价，窗口内就只会发生一次「确实值得」的换人。
            lam[max(0, f - win):min(n, f + win)] = lam_bnd
    lam[0] = 0.0

    # re-ID 半径：ID 断裂后同一个人在**相邻帧**的位移上限。
    #   注意是"一帧的位移"，不是"人有多高"。取 0.35×身高已经很宽松
    #   （30fps 下一帧移动超过 1/3 身高的舞者不存在）。半径过大 =
    #   宣布"跳到隔壁人免费"，DP 会退化成逐帧 argmax、C位 两边摇摆。
    bh_all = [_body_height(p, cfg["min_kp_conf"])
              for rec in records[:min(200, n)] for p in (rec.get("people") or [])]
    med_bh = float(np.median(bh_all)) if bh_all else 200.0
    reid_px = float(cfg.get("reid_body_frac", 0.35)) * med_bh

    path, free_at = _viterbi_focus(emit, pos, tid, lam, reid_px)

    # ---- 组装输出 ----
    primary, switch_frames = [], []
    n_reid = 0
    for i in range(n):
        k = int(path[i])
        ppl = people_at[i] or []
        focus = ppl[k] if (k < len(ppl) and np.isfinite(emit[i, k])) else None
        primary.append(focus)
        if i > 0 and path[i] != path[i - 1]:
            # ★用 DP 自己的 free 矩阵判定，不能另起一套判据 ——
            #   之前用 dist<reid_px 单独判，把 894 次真实摇摆记成"ID断裂续接"，
            #   报出"换 C位 0 次"，等于让日志替 bug 打掩护。
            fr = free_at[i]
            if fr is not None and fr[path[i - 1], path[i]]:
                n_reid += 1
            else:
                switch_frames.append(i)

    # FEATURE/GROUP 判决（沿用双阈值迟滞；它只影响构图松紧，不影响选谁）
    mode_state = ComposeState(fps, cfg)
    compose, group_boxes = [], []
    for i in range(n):
        row = emit[i]
        sc = [float(x) for x in row if np.isfinite(x)]
        compose.append(mode_state.update(sc))
        group_boxes.append(_union_box(people_at[i] or []))

    n_free = int((lam <= lam_bnd).sum())
    n_hit = sum(1 for p in primary if p is not None)
    # ★独立度量：直接测焦点在画面里的横向跳变，不依赖 DP 自己的任何判据。
    #   "换C位0次"这种自证清白的日志曾经把摇摆 bug 藏了整整一轮。
    fx = np.array([yp._person_center(p)[0] if p else np.nan for p in primary])
    jump = np.abs(np.diff(fx))
    jump = jump[np.isfinite(jump)]
    big = int((jump > reid_px).sum()) if jump.size else 0
    print(f"[subject] 全局DP选C位：{n} 帧 × 每帧最多 {K} 人（re-ID 半径 {reid_px:.0f}px）· "
          f"{len(marks)} 个换人时机（段落/硬切，占 {100.0*n_free/n:.0f}% 帧）· "
          f"λ：段内={lam_mid} 边界={lam_bnd}")
    print(f"[subject]   实际换 C位 {len(switch_frames)} 次 · "
          f"ID断裂续接 {n_reid} 次（同一人换工牌）· C位命中 {n_hit}/{n} 帧")
    print(f"[subject]   焦点横向跳变：中位 {np.median(jump) if jump.size else 0:.1f}px · "
          f"95分位 {np.percentile(jump, 95) if jump.size else 0:.1f}px · "
          f"超过 re-ID 半径的跳变 {big} 次 ← 这个数大就是在人之间摇摆")

    primary = yp.fill_primary_gaps(primary, max_gap=int(cfg["fill_gap_frames"]))
    meta = {"compose_mode": compose, "group_box": group_boxes,
            "focus_switch_frames": switch_frames}
    return primary, meta
