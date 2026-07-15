# AI 运镜系统 · 代码说明文档

本文档回答三件事：(1) 你的基本运镜标准是否已落进代码、落在哪；(2) 每个代码文件的角色（复用 / 改 / 新增）；(3) “视频运镜学习模块”是哪一块、现状如何。最后给出当前缺口的可施工补丁与配置旋钮。

贯穿全系统的两条原则：**智能在上游**（选主体、仲裁、模板）、**安全在下游**（安全框、降级、夹取永远兜底）；一切量用**归一化相对量**（秒 / 画面比例 / 覆盖率 / 角色），绝对像素只在执行那一刻翻译。

---

## 一、你的标准 → 代码落点（审计）

> 结论：大部分已覆盖，四条未真正生效。表中“落点”指实际执行该标准的函数。

| 你的标准 | 状态 | 落点 / 说明 |
| --- | --- | --- |
| 节奏强 → 合重拍 | 覆盖（尽力，非保证） | `shotplan.quantize_cuts_to_beat`（换镜吸附拍点、重大切换优先强拍）+ `camera._plan_beat_keyframes` 的 `downbeat_punch`。注意会被 `emax`、能量归一化压平，属“尽量合上”而非“一定”。 |
| 动作幅度大 → 跟随动作设计运镜 | 覆盖 | `pose_events` 出 jump/big_move → `shotplan.DEFAULT_EVENT_MAP` → extreme_wide + pull_out/follow；`camera` 跟随主体。 |
| 节奏舒缓 → 人物动作占比多；活跃 → 稍多运镜 | 部分 | `shotplan.DEFAULT_SECTION_DEFAULT`：low→wide/follow、mid→medium/follow、high→medium/push_in，方向正确。但段落标签由 `music.py` 的**本歌自归一化能量**决定，慢歌也会产生“high”段，跨歌“慢 vs 快”的绝对区分丢失。见缺口 ④。 |
| 单人：特写/中景/远景 符合裁剪 + 安全框 | 覆盖（最强项） | `camera._shot_targets`（景别→目标覆盖率/锚点）+ `_safe_box`/`_apply_safe_frame`（头胸硬约束、上半身底线、全身软目标）+ `_downgrade_shots_by_subject_size`（主体过大自动降级）。 |
| 多人：跟随主人物 / C位 | 覆盖（FEATURE） | `subject.select_subject`（C位分数 + 迟滞状态机）。GROUP 群体并集框待接（见二、camera 行）。 |
| 姿态 → 景别：全身动作→大远景/远景；上半身幅度大→中景/特写 | 覆盖 | `DEFAULT_EVENT_MAP`：jump/leap/big_move/travel→extreme_wide；level_change/floor→wide；gesture/arm_hit→medium；face→closeup。 |
| 重拍 + 旋转 优先级高 | **半缺** | 重拍已覆盖（downbeat_punch 幅度大于 beat_pulse、`prefer_downbeat_for_major`）。但 `pose_events` 会产出 **spin / extension** 两类事件，而 `shotplan.DEFAULT_EVENT_MAP` 里**没有这两个键**，事件被静默丢弃——“旋转优先级高”这条实际没生效。见缺口 ①。 |
| 轻拍 → 按旋律风格可有可无 | 部分 | `beat_pulse` 对所有拍生效、幅度按 strength 缩放；但“按风格可有可无”尚未做成风格条件（需模板 `style.rotation_usage` / `accent_profile.sparsity_per_bar`）。 |
| 景别留白：远景 头顶1/4、脚下1/6；中/特写 手臂出镜、头顶1/3~1/4 | 部分 | 中景手臂出镜已保证（`_safe_box` 的 upper 框含双腕）；但**特写的 core 框不含手腕**，举手在特写下会被裁（缺口 ②）。“头顶1/4、脚下1/6”等**具体比例未显式编码**，现状是近似居中 / 略偏上（`core_y_anchor≈0.45`），非你规定的精确留白（缺口 ③）。 |

---

## 二、逐文件角色（复用 / 改 / 新增）

| 文件 | 归类 | 说明 |
| --- | --- | --- |
| `yolo_pose.py`（单人版） | 重改 / 部分弃用 | `run_yolo_pose`（纯检测）被多人 `run_tracking` 取代；`select_primary_track` + `_merge_tracks_into_subject` 被 `subject.select_subject` 取代。保留其纯函数 `_person_center`/`_person_area`/`fill_primary_gaps`/`build_primary_records` 供 `subject` 复用。`_merge_tracks_into_subject` 多半整块作废（ByteTrack persist 已在上游解决 track 断裂）。 |
| `yolo_pose.py`（多人版）+ `registry.py` | 新增前端 | ByteTrack `model.track`，产 `tracked_keypoints.json`（带 `tracker_id`）。 |
| `subject.py` | **新增（§4）** | C位角色化桥接层：多人 records → 逐帧 C位 person（与旧 `primary_person` 同 schema）+ meta（compose_mode / group_box / 焦点切换帧）。FEATURE 模式下下游零改动。 |
| `analysis.py` | 重改 | pose 支线：records 来源换 `tracked_keypoints.json`，选主体换 `subject`；`primary_series_to_kpts` / `analyze_pose_from_kpts` 原样复用。music 支线不动（除缺口 ④ 的一处小改）。 |
| `music.py` | 复用 + 1 处必改 | 拍点/段落/能量全可用；`energy_curve_env` 需补一个跨歌绝对响度参考（缺口 ④）。 |
| `pose_events.py` | 复用 + 1 处修 | 事件机直接跑在 C位序列上；`level_change` 检测里的 `break` 应去掉（否则一支舞只记第一次重心变化）。 |
| `camera.py` | 半复用半重改 | **安全核冻结不动**：`_safe_box` / `_apply_safe_frame` / `_downgrade_shots_by_subject_size` / `_smooth_track` / emax 夹取。**强调核将被替换**：`_plan_beat_keyframes` + `_apply_beat_accents`（现有“无条件加法脉冲”）由仲裁层（§5）取代。**新增 GROUP 路径**：`_subject_geometry` 认 `compose_mode=="GROUP"` 时用 `group_box` 当 full/upper 框。 |
| `shotplan.py` | 改造 | 保留 `_enforce_min_shot` / 拍点吸附 / `_merge_dict`；补 spin/extension/focus_switch 事件映射（缺口 ①）、后续接 condition_table 融合与 GROUP 偏 wide。 |
| `transform.py` | 原样复用 | `camera_matrix` / `crop_rect` / `apply_to_points` / `effective_max_zoom`，纯几何。 |
| `render.py` / `skeleton.py` / `ffio.py` / `timeline.py` / `probe.py` | 原样复用 | 三路输出、绘骨架、编解码、统一时间轴、探测软校验。 |
| `viz.py` / `context.py` | 小改 | 景别枚举统一到 `extreme_wide`（`viz.plot_shotplan` 与 `context` 注释仍是旧的 wide/medium/upper/closeup）；`context.CameraParams` 可加 `target`/`compose_mode` 字段。 |
| `track_stats.py` | 复用（诊断） | 评估轨迹连续性的现成工具。 |
| canonical frame | 新增（§3） | `build_canonical_frame`，产 `headroom`，喂 shotplan/camera。是“加一层”，不是重写——输入正是 `camera._subject_geometry` 已算好的 `fb_*`/`bh`。 |
| 多因素仲裁 | 新增（§5） | 局部显著度 + 时间包络 + 动态预算，替换加法脉冲。 |
| 运镜反解 / 联合编码 / 风格库 | 新增（§9） | 见第三节。 |
| 模板套用 | 新增（§10） | 重标定、兼容门控。 |
| 评估体系 | 新增（§11）+ 部分复用 | `camera._metrics` 已产 `metrics.json`；测试集/主观表/风格相似度是新的。 |

---

## 三、“视频运镜学习模块”是哪一块

指**离线模板学习链路（方案 §9–§10）**，即：

```
参考视频 ─► 感知(pose+跟踪+music) ─► 运镜反解 ─► 联合编码 ─► 模板 IR ─► 风格库 ─► 套用
                                     (景别/相机运动/换镜)  (条件表+风格描述子)  (重标定+兼容门控)
```

组成与现状：

- **运镜反解 `estimate_camera_move`**：从参考视频抠出 action 序列（景别用 bbox 占 canonical 比例分桶；相机运动用 2D 相似变换近似 pan/tilt/zoom/roll；换镜用场景切分）。**现状 0%，全新增。**
- **联合编码**：condition_table（状态→动作，带样本置信度）+ style_descriptor（定长风格指纹）+ accent_profile（局部对比→强调幅度的响应曲线）。**现状 0%。**
- **风格库 + 套用**：描述子聚类成风格；套用时重标定力度、兼容门控筛掉不合适组合。**现状 0%。**

**关键区分：** 你现有的全部代码（`camera` / `shotplan` / `subject` / `analysis` …）属于**在线生成 / 执行**，用的是**人手写规则**（`DEFAULT_EVENT_MAP`、`DEFAULT_SECTION_DEFAULT`）。学习模块的作用是**用数据把这张规则表学出来**，替代手写。它是里程碑阶段 4，依赖阶段 0–3（评估、单视频、归一化、多人）先就位。当前不要与规则引擎混谈。

> 单参考的坦白：condition_table 的状态空间（section×beat_phase×pose_event×regime）叉乘上百桶，一条参考填不满几个，多数回退默认；所以单参考模板真正的载体是 **style_descriptor + accent_profile**，condition_table 名义核心、实则近空壳。需靠同风格多片聚合才有信息量。

---

## 四、当前缺口与补丁

### 缺口 ①（高优先，直接违背“旋转优先级高”）：spin / extension 事件被丢弃

`pose_events.detect_events` 会产出 `spin` / `extension`，但 `shotplan.DEFAULT_EVENT_MAP` 无对应键，事件被跳过。补两行即可让旋转事件被接上、且高优先、且真的产生旋转（`move="roll"` 在 `camera` 已实现）：

```python
# shotplan.py · DEFAULT_EVENT_MAP 内补充
# 旋转：优先级设高（近 jump=9，高于 gesture=5），move=roll 让相机真的转
"spin":      {"shot": "medium", "move": "roll",    "priority": 7, "dur_sec": 0.9},
# 伸展：上半身表现力动作，给中景 + 轻推
"extension": {"shot": "medium", "move": "push_in", "priority": 6, "dur_sec": 0.7},
```

### 缺口 ②：特写不保证手臂出镜

`camera._subject_geometry` 的头胸核心点集 `core_pts` 只含 头(鼻/眼/耳) + 双肩，不含手腕；特写走 core 框，举手会被裁。修法：特写档单独把双腕纳入“必须出镜”的框（不要污染 core 的“头胸居中”语义）。在 `_shot_targets` 的 closeup 分支里，用一个含腕的框决定 `content_h`，或在 `_safe_box` 增加一个 `kind="closeup"` 专用框 = 头 + 肩 + 双腕。改动局限在 camera 一处，安全核其余不动。

### 缺口 ③：留白比例（头顶1/4、脚下1/6）未显式编码

现状是近似居中/略偏上。要精确实现你的规格，在 `_shot_targets` 里把“留白比例”写成配置并据此算 `content_h` 与竖直锚点。以远景为例，正文占 `1 - 1/4 - 1/6 = 7/12` 画面高、头顶落在 `0.25·view_h`：

```python
# camera.py · _shot_targets 的 wide 分支示意（比例走 config，便于按你的标准调）
head_frac = cover_cfg.get("wide_head_top_frac", 0.25)   # 头顶留白
foot_frac = cover_cfg.get("wide_foot_bot_frac", 1/6)    # 脚下留白
body_frac = 1.0 - head_frac - foot_frac                 # 正文占比 = 7/12
content_h = bh                                          # 全身高（源像素）
view_h_target = content_h / body_frac                   # 反推所需窗高
z_t = base_h / view_h_target                            # 覆盖率→zoom
# 竖直锚点：让头顶落在窗内 head_frac 处（构图偏上），由 safe_frame 的 y_anchor 落实
```

中景/特写同理（头顶留 1/3~1/4，底边到腰/胸）。把这些比例集中到 `camera.shot_coverage` 配置块，你就能不改代码按标准微调。

### 缺口 ④：慢 / 快歌跨片区分被能量自归一化吃掉

`music.py` 的 `energy_curve_env` 用 `rms / rms.max()` 归一，慢歌与燥歌各自铺满 `[0,1]`，段落 low/mid/high 变成“相对本歌”，跨歌绝对响度丢失——正是“舒缓 vs 活跃”那条规则的依据。修法：归一化前留一个绝对参考标量并输出，供 shotplan 判“这整首是安静的歌”。

```python
# music.py · energy_curve_env 内，归一化之前
abs_rms_ref = float(np.median(rms))     # 绝对响度参考（未归一化）
# ... 之后照旧 rms = rms / rms.max()
# analyze_music 的返回里带上 "abs_loudness": round(abs_rms_ref, 6)
# shotplan 侧：整首 abs_loudness 低于阈值 → 全局偏 follow/wide、抑制 push_in/脉冲
```

---

## 五、配置旋钮（按你的标准调，不改代码）

- **景别留白**：`camera.shot_coverage`（每档覆盖率；补 `*_head_top_frac` / `*_foot_bot_frac`）。
- **旋转/事件优先级**：`shotplan.event_map`（覆盖 spin/extension 的 priority、move）。
- **拍点强调力度与不应期**：`camera.templates.beat_pulse` / `downbeat_punch`；未来 `accent_profile.sparsity_per_bar`（每小节最多几下）。
- **段落基调**：`shotplan.section_default`（low/mid/high 的 shot/move）；接 `abs_loudness` 后可做“安静歌全局收敛”。
- **C位选择偏好**：`analysis.pose.subject.weights`（center/scale/motion/frontal）、`hysteresis`（换人快慢）、`compose`（FEATURE/GROUP 阈值）。
- **安全框**：`camera.safe_frame`（margin、core_center_pull、fullbody_events、downgrade_ratio）。

---

## 六、施工优先级（与里程碑对齐）

1. **补缺口 ①**（两行，立即让旋转标准生效）与 **④**（一处，救回慢/快歌区分）——投入最小、直接对上你的标准。
2. **camera 劈“安全核 / 强调核”** + **仲裁层（§5）** 替换加法脉冲——阶段 1“治单视频”的主体。
3. **缺口 ②③**（留白精确化）——与仲裁同批调，肉眼直接可见。
4. **canonical frame（§3）** → **GROUP 路径** → **学习模块（§9–10）**，按里程碑顺序推进。
