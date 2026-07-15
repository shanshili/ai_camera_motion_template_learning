# AI 运镜系统 · 方案流程与原理说明

本文对应当前代码库的实际实现，逐环节说明**做什么、为什么这样做、用什么公式**，并标注对应代码位置。

---

## 0. 两条总原则

**原则一：归一化参考系。** 模板只存与视频无关的相对量；绝对像素/帧只在套用那一刻翻译：

$
n_{\text{frames}} = \mathrm{round}(t_{\text{sec}} \cdot \mathrm{fps}), \qquad
p_{\text{px}} = f_{\text{frac}} \cdot D_{\text{dim}}
$

横向量乘宽、纵向量乘高，天然兼容 16:9 与 9:16。景别用**覆盖率**而非倍数，主体身份用**角色（C位）**而非 `track_id`。

**原则二：智能在上游，安全在下游。** 决策智能（选主体、分镜、模板）在上游；执行层（安全框、限速、夹取）保持"笨而可靠"，是永远兜底的护栏。无论上游算出什么，最终都过这层。

---

## 1. 系统总览

```
【离线：模板学习】
参考视频 ─► 感知(pose+跟踪+music) ─► 运镜反解(景别/换镜/节奏) ─► 模板 IR (θ) ─► 预览
                                                                    │
【在线：运镜生成】                                                    ▼
用户视频 ─► 感知 ─► 主体选择(C位) ─► shotplan(读模板出分镜) ─► camera(安全执行) ─► render
```

| 环节 | 代码 | 输入 → 输出 |
| --- | --- | --- |
| 多人骨架 | `multi_person_tracking/run_tracking.py` | 视频 → `tracked_keypoints.json` |
| 音乐分析 | `pipeline_3/music.py` | 音频 → 拍点/能量/段落/响度 |
| 姿态事件 | `pipeline_3/pose_events.py` | C位关键点 → 事件序列 |
| 主体选择 | `pipeline_3/subject.py` | 多人 → C位 + FEATURE/GROUP |
| 模板契约 | `pipeline_3/template.py` | 模板 IR 读写、兼容判别 |
| 模板学习 | `pipeline_3/learn_template.py` | 参考视频 → `template.json` |
| 分镜 | `pipeline_3/stages/shotplan.py` | 模板+状态 → 分镜表 |
| 相机 | `pipeline_3/stages/camera.py` | 分镜 → 逐帧 zoom/center/rot |
| 几何 | `pipeline_3/transform.py` | 相机参数 → 仿射矩阵 |
| 渲染 | `pipeline_3/stages/render.py` | 矩阵 → 成片 |

---

## 2. 感知层

### 2.1 多人骨架（`run_tracking.py`）

YOLO-pose 逐帧检测 + ByteTrack 跨帧关联，输出每帧 `people[]`，每人带 `tracker_id`、`box_xyxy`、17 个 COCO 关键点 `(x, y, conf)`。

**原理要点：不要求 ID 全程稳定。** 舞蹈中快速换位/交叉遮挡必然导致 ID 跳变，下游 C位角色化用"位置最近邻"续任，对 ID 抖动免疫（见 §4.2）。因此 tracker 配置用 loose 参数（`track_buffer=90`，更长的容忍期）。

### 2.2 音乐分析（`music.py`）

**起音包络（onset envelope）：** 对 STFT 幅度谱取对数后做正向差分求谱通量，再减去局部均值以突出瞬态：

$
\mathrm{flux}(t) = \sum_{k} \max\!\big(0,\; \log(1+|X_{k,t}|) - \log(1+|X_{k,t-1}|)\big)
$

$
\mathrm{env}(t) = \max\!\big(0,\; \mathrm{flux}(t) - \overline{\mathrm{flux}}_{[t-4,\,t+4]}\big)
$

**速度估计：** 对包络做自相关，配一个对数域高斯先验（抑制倍频/半频误判，中心 $bpm_0=120$）：

$
\mathrm{prior}(bpm) = \exp\!\Big(-\tfrac{1}{2}\Big(\tfrac{\log_2(bpm/bpm_0)}{0.7}\Big)^{2}\Big),
\qquad
P^{*} = \arg\max_{\text{lag}} \; \mathrm{AC}(\text{lag}) \cdot \mathrm{prior}(bpm(\text{lag}))
$

峰值处用抛物线插值取亚采样精度。

**拍点跟踪（Ellis 动态规划）：** 在"包络强"与"间隔接近周期 $P$"之间做全局最优权衡：

$
C(t) = \mathrm{env}(t) + \max_{v \in [t-2P,\; t-P/2]} \Big[\, C(v) - \lambda \big(\log\tfrac{t-v}{P}\big)^{2} \Big]
$

其中 $\lambda=100$ 为节奏刚性。回溯 $\arg\max$ 得到全局最优拍序列。这比"取局部峰"稳健得多——它保证拍与拍的**间隔一致性**。

**强拍：** 在 $m=4$ 的相位里选包络能量和最大的那个相位：

$
\phi^{*} = \arg\max_{\phi \in [0,m)} \sum_{j} \mathrm{env}\big(b_{\phi + jm}\big)
$

**段落切分（Foote novelty）：** 构造色度自相似矩阵 $S = C C^{\top}$，沿对角线卷一个棋盘核 $K$（对角块 $+1$、反对角块 $-1$，加 Hann 窗）：

$
\mathrm{nov}(i) = \sum_{u,v} S[i{-}L{:}i{+}L,\; i{-}L{:}i{+}L]_{u,v}\cdot K_{u,v}
$

峰值即段落边界。段内按平均能量打 `low/mid/high` 标签。

**绝对响度（关键设计）：** 能量曲线按本歌最大值归一化会让慢歌和燥歌**各自铺满 $[0,1]$**，跨歌的"安静 vs 吵闹"信息丢失。因此在归一化**之前**另存一个绝对参考：

$
L_{\text{abs}} = \mathrm{median}\big(\mathrm{RMS}(t)\big) \quad (\text{未归一化})
$

下游据此判"这整首是安静的歌"，全局收敛运镜（`shotplan` 的 `quiet` 分支）。

### 2.3 姿态事件（`pose_events.py`）

从 C位的 17 点算逐帧特征：质心、身高 $h_b$、肩宽、肢体伸展度、bbox。**运动能量按身高归一**（消除远近尺度影响）：

$
m(t) = \frac{\|\mathbf{c}(t) - \mathbf{c}(t-1)\|_2}{h_b} \cdot \mathrm{fps}
$

事件检测（均带最小间隔去重）：

| 事件 | 判据 | 直觉 |
| --- | --- | --- |
| `jump` | $(\tilde{c}_y - c_y)/h_b > 0.12$ 的峰，$\tilde{c}_y$ 为 1 秒滑动中位基线 | 质心显著高于基线（图像 y 向下） |
| `extension` | 伸展度显著峰且 $> 1.15\times$ 中位 | 肢体外张 |
| `spin` | 肩投影宽持续 $< 0.55\times$ 中位 | 侧身/旋转时肩宽收窄 |
| `freeze` | $m(t) < 0.05$ 持续 $\ge 0.3$s | 定格 |
| `level_change` | 长窗基线位移 $> 0.25 h_b$ | 蹲下/起身 |

**One Euro 平滑**（自适应低通，慢动强滤、快动少延迟）：

$
f_c = f_{\min} + \beta |\dot{\hat{x}}|, \qquad
\tau = \frac{1}{2\pi f_c}, \qquad
\alpha = \frac{1}{1 + \tau/T_e}, \qquad
\hat{x}_t = \alpha x_t + (1-\alpha)\hat{x}_{t-1}
$

截止频率随速度上升 → 快速运动时几乎不滤（低延迟），静止时强滤（去抖）。

---

## 3. 主体选择层（`subject.py`）

**核心思想：锁定"C位"这个角色，允许换人但换得慢、换得稳。**

### 3.1 C位分数

对每帧每人打分：

$
C_i(t) = w_p P_i + w_z Z_i + w_m M_i + w_f F_i + w_e E_i
$

| 项 | 定义 | 说明 |
| --- | --- | --- |
| $P_i$ 居中度 | $\mathrm{clip}\big(1 - \frac{\|\mathbf{c}_i - \mathbf{g}\|}{0.5\sqrt{W^2+H^2}},\,0,\,1\big)$ | $\mathbf{g}$ 为全体质心；单人时恒为 1 |
| $Z_i$ 相对尺度 | $A_i / \max_j A_j$ | 近大远小 |
| $M_i$ 相对运动 | $r/(1+r),\ \ r = m_i/\mathrm{median}_j(m_j)$ | **别人静他动 → 高**。★必须与 P/Z/F 同量纲 $[0,1]$：旧写法 $\mathrm{clip}(r,0,3)$ 上限为 3、配权重 0.8 → 单项浮动 2.4，比其它三项之和还大；而舞蹈中瞬时运动逐帧剧烈波动，直接导致排序不停翻转、C位乱跳 |
| $F_i$ 正面度 | $\mathrm{clip}\big(\frac{\|x_{ls}-x_{rs}\|}{0.35 h_b},\,0,\,1\big)$ | 侧身时肩宽收窄 |
| $E_i$ 事件加成 | 预留（默认 0） | 避免"选焦点"与"算事件"互为因果 |

权重是模板的一部分——不同风格"看谁"的偏好不同。

### 3.2 迟滞状态机（防乱跳）

现任 $\mathrm{inc}$ 由**上一帧焦点位置的最近邻**续任（不绑 `track_id`）：

$
\mathrm{inc} = \arg\min_{i} \|\mathbf{c}_i - \mathbf{p}_{\text{focus}}\|, \quad \text{s.t. } \|\mathbf{c}_i - \mathbf{p}_{\text{focus}}\| \le R
$

打分时给现任加成 $b$，换人需**同时**满足"够强"与"够久"：

$
\text{switch} \iff \Big[\big(C_{\text{top}} - (C_{\text{inc}} + b)\big) > \delta\Big] \text{ 连续成立 } \ge \tau \cdot \mathrm{fps} \text{ 帧}
$

默认 $b=0.35,\ \delta=0.25,\ \tau=1.0$s。换焦点时注入 `focus_switch` 事件，用慢速 recenter 平滑转移。

**★段落锁定（治「C位乱聚焦」的主力）：** 实拍规律是——**换唱段才换 C位，切景才短暂给别人**；一个唱段内即便 C位 移动，也不该换人。因此只在**段落边界 / 硬切**附近 $\pm w$ 内允许换人，其余帧只续任现任：

$
\text{allow}(t) = \bigvee_{f \in \{\text{段落起点}\} \cup \{\text{硬切}\}} \mathbb{1}\big[|t - f| \le w \cdot \mathrm{fps}\big]
$

这把换人机会从"每帧"降到"每段一次"。现任丢失后的**重捕获**不受此限——那不是换人，是找回。

> **实测**（11 人群舞 / 6 段 / 运动剧烈波动）：$M$ 量纲修复前 **38 次**换焦点/65s → 修复后 8 次 → 加段落锁定后 **2 次**，且都落在段落边界。

### 3.3 FEATURE / GROUP 判决

注意力集中度：

$
\mathrm{conc} = \frac{C_{(1)} - C_{(2)}}{C_{(1)} + \varepsilon}
$

**双阈值迟滞**（防逐帧翻转）：处于 FEATURE 时需 $\mathrm{conc} \le \theta - r$ 才转 GROUP；处于 GROUP 时需 $\mathrm{conc} > \theta + r$ 才回 FEATURE；且需持续 $\ge$ `hold_sec`。默认 $\theta=0.35,\ r=0.10$。

- **FEATURE**：框 C位 → 走单人机器。
- **GROUP**：记录群体并集框，`shotplan` 据此限制景别下限。
- 单人：$n=1$ 时恒等退化，零成本兼容。

### 3.4 短间隙插值

C位缺失 $\le 15$ 帧的空洞两端线性插值（`fill_primary_gaps`），保证下游几何序列无空洞。两端必须都有真实主体，序列首尾的悬空空洞不补。

---

## 4. 模板 IR（`template.py`）

模板 = **归一化参考系里的条件策略参数**，不是任何绝对曲线。

```jsonc
{
  "meta": {
    "orientation": "landscape",        // 朝向限定（硬否决用）
    "n_regime": "group",               // 单人/群舞
    "bpm": 110.94, "bpm_tolerance": 25.0,
    "people_count": {"min":1,"max":15,"median":6},   // 出镜人数区间
    "c_exposure": {"full_body":0.57,"half_body":0.23,"head":0.21},  // C位出镜程度
    "shot_hist": {...}                 // 景别分布
  },
  "style": {
    "cut_rhythm_sec": 2.96,            // 换镜节奏（秒）
    "group_min_shot": null,            // GROUP 帧景别下限（学出）
    "quantize_to_beat": true,
    "quiet_loudness_th": 0.08
  },
  "section_shot_dist": {               // ★段落→景别分布（还原景别变化的关键）
    "high": {"wide":0.41,"medium":0.23,"closeup":0.21,"extreme_wide":0.15}
  },
  "section_default": {...},            // 众数兜底
  "event_map": {"spin": {"shot":"medium","move":"roll","priority":7,"dur_sec":0.9}, ...},
  "shot_coverage": {...},              // 景别留白规格
  "accent": {...},                     // 强调层
  "style_descriptor": [...]            // 定长风格指纹
}
```

### 4.1 兼容判别

**先硬否决，再算软距离**——这点很重要：人数 regime 与朝向不匹配是**类别否决**，不该被 BPM 相似度稀释。

硬否决（返回 0）：
- 朝向不一致（横向模板套竖屏）——构图/留白规格根本不同；
- 群舞模板套独舞——`target=group` 的动作在独舞上物理不存在；
- 目标人数中位超出模板见过的范围太远。

软距离：

$
\mathrm{compat}(T,V) = \exp\!\Big(-\sum_k \lambda_k D_k\Big)
$

$
D_{\text{bpm}} = \frac{\max\big(0,\; |bpm_V - bpm_T| - \text{tol}\big)}{bpm_T},
\qquad
D_{\text{people}} = \frac{|\,\tilde{n}_V - \tilde{n}_T\,|}{\tilde{n}_T}
$

容差内视为完全匹配。分数 $<0.3$ 给告警但不硬拦——决定权留给使用者。

---

## 5. 模板学习（`learn_template.py`）

### 5.1 换镜检测

相邻帧 HSV 二维直方图（H×S = 32×32）的相关系数骤降即硬切：

$
\mathrm{corr}(h_{t-1}, h_t) < 0.6 \;\Rightarrow\; \text{cut at } t
$

### 5.2 景别识别（按 C位露出/遮挡）

**核心区分：「关节没检测到」$\ne$「关节在画面外」。**

- 被别的舞者挡住 = 遮挡（漏检）→ 画面其实是远景；
- 被画幅切掉 = 取景意图 → 这才是景别信号。

只有后者能反映景别。因此先看 bbox 是否贴到画面边缘：

$
\text{若 } y_1 < H - \epsilon: \quad \text{全身在画面内} \Rightarrow
\begin{cases}
\texttt{extreme\_wide}, & r < 0.45\\
\texttt{wide}, & r \ge 0.45
\end{cases}
\qquad r = \frac{y_1 - y_0}{H}
$

$
\text{若 } y_1 \ge H - \epsilon: \quad \text{被下边缘裁切} \Rightarrow
\begin{cases}
\texttt{wide}, & \text{踝或膝可见}\\
\texttt{medium}, & \text{髋可见（切腿）}\\
\texttt{closeup}, & \text{仅肩/头（切到胸）}
\end{cases}
$

同时统计 C位出镜程度 `c_exposure`（full_body / half_body / head 占比）。**C位参考比重最大**，其他人只进人数统计。

### 5.3 换镜节奏（含病态值夹取）

$
T_{\text{cut}} = \mathrm{clip}\Big(\frac{\mathrm{median}(\text{段长})}{\mathrm{fps}},\; 0.6,\; 4.0\Big)
$

**为什么要夹：** 若参考是固定机位未剪辑视频（换镜数 $<3$），段长中位数 $\approx$ 全片长度，直接采用会让 `min_shot` 大到把整片并成一段。此时回退默认值。

### 5.4 卡点命中率

$
\mathrm{sync} = \frac{\big|\{c \in \text{cuts} : \min_b |b - c| \le 0.15\,\mathrm{fps}\}\big|}{|\text{cuts}|}
$

参考本身卡点（$\mathrm{sync} \ge 0.4$）才开启 `quantize_to_beat`。

### 5.5 景别分布（关键设计）

$
\mathrm{dist}[\ell][s] = \frac{\text{段落标签 } \ell \text{ 中景别 } s \text{ 的帧数}}{\text{标签 } \ell \text{ 的总帧数}}
$

**为什么保留分布而非众数：** 只取众数会把"21% 特写 + 23% 中景 + 41% 远景"塌缩成"远景"，模板表达力退化成每个段落标签一个景别，MV 的景别变化全部丢失。

### 5.6 GROUP 下限自学

$
\texttt{group\_min\_shot} =
\begin{cases}
\texttt{null}, & p_{\text{closeup}} + p_{\text{medium}} > 0.25\\
\texttt{"wide"}, & \text{否则}
\end{cases}
$

参考自己（同为群舞）若大量用中景/特写，套用时就不该把 GROUP 帧一律压成远景——否则与学到的分布自相矛盾。

### 5.7 诚实边界

- **只反解可复现的构图意图**：景别分布、换镜节奏、卡点命中、段落倾向。
- **不做 homography 反解真实相机运动**：裁剪相机无法复现真 dolly（主体不变大、背景变），学出来也执行不了。
- **单参考的条件表近乎空壳**：状态叉乘上百桶，一条参考填不满几个。风格主要由 `section_shot_dist` + `style_descriptor` + `accent` 承载。

---

## 6. 分镜层（`shotplan.py`）

### 6.1 镜头网格

按学到的换镜节奏切网格，边界吸附拍点：

$
\text{step} = \mathrm{round}(T_{\text{cut}} \cdot \mathrm{fps}), \qquad
b_k \leftarrow \arg\min_{\beta \in \text{beats},\, |\beta - b_k| \le w} |\beta - b_k|
$

重大切换（优先级跨度 $\ge 2$）优先吸附**强拍**。

### 6.2 模板与本片事件的分工（核心设计）

**模板与本片事件不是竞争关系，而是"多少"与"哪一段"的分工：**

| | 回答的问题 | 来源 |
| --- | --- | --- |
| **模板** | **多少**：68% 远景、8% 中景、每 3.6s 换镜 | 参考视频的编排意图 |
| **本片事件** | **哪一段**：这 3 秒有大跳（须给松）、那 3 秒是上身伸展（适合给近） | 目标视频的内容 |

让事件以优先级去覆盖模板，等于让"哪一段"否决"多少"——事件往往覆盖 80%+ 的帧，模板的景别分布会被整个抹掉，表现为"模板没效果、全片一个景别"。

**近景亲和度**（本片事件作为"参考"）：复用 `event_map` 里已有的 `shot` 字段作为信号

$
\mathrm{aff}(g) = \sum_{e \,\cap\, g} A\big(\mathrm{shot}(e)\big)\cdot\big(0.5 + \iota_e\big)\cdot \frac{|e \cap g|}{|g|},
\qquad
A = \{\texttt{closeup}{:}\,2,\; \texttt{medium}{:}\,1,\; \texttt{wide}{:}\,{-}1,\; \texttt{extreme\_wide}{:}\,{-}2\}
$

**几何兜底**（本片事件作为"保障"）：`owns_shot=True` 的事件（jump/leap/big_move/travel/level_change/floor）在几何上**必须**更松的景别才装得下，否则出画。这类段无论配额如何都强制给该景别（多个强制事件取最松）。

### 6.3 配额分配

对每个段落标签：先用最大余数法把分布换算成整数配额

$
c_s = \Big\lfloor p_s \cdot m \Big\rfloor + \text{（按余数从大到小补足至 } \textstyle\sum_s c_s = m\text{）}
$

**几何强制段先落位，并从配额中扣除**（否则强制段等于外挂，全片比例会被挤偏），剩余配额归一到自由段；自由段按 $\mathrm{aff}$ 从高到低排序，配额**从近到远**依次发下去——特写发给最适合特写的那几段。

事件层随后只贡献 `move`（同一镜头内可推/摇/滚），不再改景别。

> **实测**（1800 帧，模板目标 wide 0.40 / ew 0.30 / medium 0.20 / closeup 0.10）：
> 全片实际 **0.40 / 0.30 / 0.20 / 0.10**（比例完全还原）；
> 上身表现区 medium 0.60 + closeup 0.30；大跳区 extreme_wide 0.90（强制）；无事件区 wide 1.00。

### 6.4 后处理

- **防碎切** `_enforce_min_shot`：★只对**景别**施加最短时长 $T_{\text{cut}}\cdot\mathrm{fps}$。运镜变化不是换镜——一个镜头内部本就可以推/摇/滚，若把 min_shot 套在 (景别,运镜) 组合上，事件驱动的运镜变化（~25 帧）会被整段并掉。
- **GROUP 下限**：`group_min_shot` 非空时，GROUP 帧的景别不得比它更近。
- **安静歌收敛**：$L_{\text{abs}} < \theta_{\text{quiet}}$ 时全局偏 follow/wide、抑制 push_in ——对应"舒缓的歌人物动作占比多"。

---

## 7. 相机层（`camera.py`）· 安全执行

### 7.1 基础几何（`transform.py`）

**基准视窗**（$z=1$ 时源画面里能容纳输出比例的最大裁剪窗）：

$
\text{若 } \frac{W_s}{H_s} \ge \frac{W_o}{H_o}: \quad H_{\text{base}} = H_s,\; W_{\text{base}} = H_s\cdot\frac{W_o}{H_o}
\qquad\text{否则}\qquad
W_{\text{base}} = W_s,\; H_{\text{base}} = \frac{W_s}{W_o/H_o}
$

**有效最大缩放**（画质预算）：

$
z_{\max} =
\begin{cases}
z_{\text{cfg}}, & \texttt{allow\_upscale} = \text{true}\$4pt]
\min\Big(z_{\text{cfg}},\; \min\big(\tfrac{W_{\text{base}}}{W_o}, \tfrac{H_{\text{base}}}{H_o}\big)\Big), & \text{否则}
\end{cases}
$

例：$1920\times1080$ 源 → $1280\times720$ 成片，画质预算 $=1.5$，故 $z_{\max}=\min(1.45, 1.5)=1.45$。**这是全片景别的总闸门**（见 §7.6 诊断）。

**裁剪窗与仿射矩阵**：$W_v = W_{\text{base}}/z,\; H_v = H_{\text{base}}/z$，

$
\mathbf{p}_{\text{out}} = S \cdot R \cdot (\mathbf{p}_{\text{src}} - \mathbf{r}_c) + \mathbf{o}_c,
\qquad
M = \begin{bmatrix} s_x\cos\theta & s_x\sin\theta & t_x \\ -s_y\sin\theta & s_y\cos\theta & t_y \end{bmatrix}
$

渲染帧与骨架点**共用同一个 $M$**，保证绝不错位。

### 7.2 景别 → 目标 zoom

覆盖率定义（内容高 / 裁剪窗高）：

$
\mathrm{cover} = \frac{h_{\text{content}}}{H_v} \;\Longrightarrow\;
\boxed{\;z = \mathrm{cover} \cdot \frac{H_{\text{base}}}{h_{\text{content}}}\;}
$

各景别的 $h_{\text{content}}$ 取不同身体段（`span`）：`full`=全身、`upper`=髋以上、`chest`=胸以上。竖直锚点让头顶落在窗内 $f_{\text{head}}$ 处：

$
c_y = y_{\text{top}} + (0.5 - f_{\text{head}}) \cdot H_v
$

即"头顶留白 = $f_{\text{head}}$"（远景 1/4、脚下 1/6；中景/特写 1/3~1/4）。

### 7.3 安全框（分层硬约束）

**装框最大 zoom**：要求裁剪窗能整个包住框 $(w_b, h_b)$：

$
z \le \min\Big(\frac{W_{\text{base}}}{w_b},\; \frac{H_{\text{base}}}{h_b}\Big)
$

**三层框**：`core`（头+双肩+上胸）、`upper`（+双髋+双腕）、`full`（+膝+踝）。

**按景别决定次级约束（关键设计）：**

$
\text{kind} =
\begin{cases}
\texttt{full}, & \text{地面动作帧（安全兜底，最高优先）}\\
\texttt{full}, & \text{shot} \in \{\texttt{wide},\texttt{extreme\_wide}\}\\
\texttt{upper}, & \text{shot} = \texttt{medium} \quad(\text{允许切腿})\\
\texttt{None}, & \text{shot} = \texttt{closeup} \quad(\text{允许切到胸})
\end{cases}
$

$
z_{\text{new}} =
\begin{cases}
\min(z_t,\, z_{\text{core}},\, z_{\text{upper}},\, z_{\text{full}}), & \text{kind}=\texttt{full}\\
\min(z_t,\, z_{\text{core}},\, z_{\text{upper}}), & \text{kind}=\texttt{upper}\\
\min(z_t,\, z_{\text{core}}), & \text{kind}=\texttt{None}
\end{cases}
$

> **为什么必须按景别分层：** 若无条件夹 $z_{\text{full}}$，等于"任何景别都保证全身入画"——中景要切腿、特写要切到胸，几何上永远不可能发生，全片只能是远景。头胸 `core` 始终是硬约束，任何景别都不破。

**构图锚点与"必须包住"解耦：** `core` 框负责硬约束，锚点用**脖子**（双肩中点）。核心框是"头顶→上胸"，其几何中心天然偏头部，拿它对齐锚点会让整体偏上。

$
\mathbf{c}_{\text{cam}} = \mathbf{a}_{\text{neck}} + \mathbf{V}\odot(0.5 - \mathbf{f}_{\text{anchor}})
$

### 7.4 尺寸降级

段内主体最大高度占比 $r = \max(h_{\text{bbox}})/H_{\text{base}}$（**分母必须是 $H_{\text{base}}$ 而非 $H_o$**——两者单位不同，误用会让降级几乎不触发）：

$
\text{allowed} =
\begin{cases}
\texttt{wide}, & r \ge 0.82\\
\texttt{medium}, & r \ge 0.68\\
\texttt{closeup}, & \text{否则}
\end{cases}
$

当前景别比 allowed 更近则降级——从源头避免"人很大却给特写导致出画"。

### 7.5 平滑

**死区**（目标在安全区内则中心不动）：

$
c_{\text{des}} =
\begin{cases}
c_{\text{tgt}} - \mathrm{sign}(\Delta)\cdot \ell, & |\Delta| > \ell\\
c_{\text{prev}}, & \text{否则}
\end{cases}
\qquad \ell_x = d_x W_v,\;\; \ell_y = d_y H_v
$

**限速**：$|c_t - c_{t-1}| \le \Delta_{\max}$（默认 32 px/帧）。

**One Euro**：同 §2.3 公式，对 center/zoom/rot 分别滤波。

**★硬切**：裁剪相机没有真多机位，"切"即构图瞬变。在标记了 `cut` 的段首帧重置滤波器状态并令 $c_{\text{prev}} \leftarrow c_{\text{tgt}}$，使死区/限速自然不触发、One Euro 直出目标——裁剪窗瞬间跳到新构图。不这样做的话，限速（32 px/帧）会把 500 px 的构图跳变抹成 16 帧的缓慢推移，表现为"全是移动运镜、一个硬切也没有"。是否用硬切由 `style.hard_cut` 决定（学习时按参考的换镜数判定）。

**片头防漂移**：`cx_prev` 用第 0 帧真实目标初始化，而非画面中心——否则死区+限速会从画面中心缓慢爬向主体，叠加 One Euro 首次大位移的速度估计误差，表现为开头几秒明显晃动。

**平滑后硬校验**：平滑可能又把安全框顶出去，故再做一次夹取；但**只做安全区间夹取，不再主动居中**，且纠偏本身也过限速——否则每帧无条件拉 60% 会绕开限速，把镜头"焊"到新位置。

### 7.6 强调层与诊断

拍点脉冲叠加到 zoom（`beat_pulse` 平滑升降、`downbeat_punch` 快起慢落 + 运动模糊），叠加后再过一次安全框夹取。

**诊断（归因）：**

$
\mathrm{sat}_{\text{emax}} = \frac{|\{t : z_{\text{want}}(t) > z_{\max}\}|}{n},
\qquad
\mathrm{sat}_{\text{safe}} = \frac{\big|\{t : z_{\text{safe}}(t) < \min(z_{\text{want}}(t), z_{\max})\}\big|}{n}
$

第二式**必须与 $\min(z_{\text{want}}, z_{\max})$ 比**——因为安全框内部也把 $z$ 夹到 $z_{\max}$，直接与 $z_{\text{want}}$ 比会把 emax 的账算到安全框头上。

### 7.7 指标

| 指标 | 含义 | 读法 |
| --- | --- | --- |
| `head_chest_in_rate` | 头胸核心在画面内比率 | **护栏，应 ≈1.0** |
| `upper_body_in_rate` | 上半身在画面内比率 | 护栏 |
| `fullbody_in_rate_on_masked` | 地面动作帧的全身命中 | 护栏 |
| `subject_in_frame_rate` | 整个 bbox 在画面内比率 | **中景/特写时必然下降**（本就要切腿），不是缺陷 |
| `center_jerk95` / `zoom_jerk95` | 三阶差分 95 分位 | 抖动护栏；低 ≠ 好看 |

指标只作护栏，不作优化目标——低抖动也可能极其无聊。

---

## 8. 渲染层（`render.py`）

对每帧做 `warpAffine`；方向性运动模糊按裁剪中心位移量触发：

$
\text{mag} = \|\Delta \mathbf{c}\|_2 \cdot \text{strength}, \qquad
k = \mathrm{clip}(\mathrm{round}(\text{mag}/3),\, 3,\, 15)
$

沿主导轴（$|\Delta x|$ vs $|\Delta y|$）取线性核。底部可追加音乐能量时间轴（`np.vstack`，不遮挡主画面）。

---

## 9. 模板预览（`preview.py`）

用参考视频**自己的真实骨架 + 真实音乐**，把学到的模板套上去渲染，长度 = 输入视频长度。画面：全体骨架淡显 + C位加粗，底部时间轴标出换镜边界、逐段景别色带、姿态事件、拍点。

这是一条**不依赖 YOLO/真视频的端到端自测路径**，也是"这个模板套到这条视频会怎么运镜"的直观答案。

---

## 10. 边界与不可实现

1. **真实推轨/摇臂/视差**：单源裁剪永远无法露出画外内容，只能逼近构图意图。
2. **源分辨率不足时的近景**：$1920$ 源里 $350$px 的人切特写需 $z \approx 10$，等于把 $\sim130$px 拉到 720p——采样定律拉不回。
3. **"任意模板套任意视频都好看"**：数学上不成立，只能靠兼容门控筛掉不合适组合。
4. **剪辑意图的具体时刻**：只能还原分布与节奏，无法预测"这一刀为什么切特写"。
5. **单参考的条件表**：状态空间叉乘上百桶，一条参考填不满几个。
6. **VLM 直出逐帧精确参数**：不可靠，只做高层风格标签。

---

## 附：一句话收束

- 模板 = 归一化参考系里的**条件策略参数**（角色 / 相对时间 / 相对空间 / 景别分布），不是绝对曲线。
- 智能在上游（选主体、分镜、模板），安全在下游（安全框、降级、夹取永远兜底）。
- 学习只反解**可复现**的东西；还原**分布与节奏**，不假装预测剪辑意图。
