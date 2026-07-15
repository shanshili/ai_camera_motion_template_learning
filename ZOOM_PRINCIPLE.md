# 运镜学习原理 · zoom 倍数是怎么定出来的

以 1920×1080 源 → 1280×720 成片、源画面里一个身高 350px 的舞者为例，逐步走一遍真实数值。

---

## 一句话

\[
\boxed{\;z \;=\; \underbrace{\mathrm{cover}}_{\text{模板学的}} \times \frac{\;\underbrace{H_{\text{base}}}_{\text{画幅几何}}\;}{\;\underbrace{h_{\text{content}}}_{\text{C位骨架逐帧算}}\;}\;}
\qquad\text{然后}\qquad
z \leftarrow \min\big(z,\; z_{\text{safe}},\; z_{\max}\big)
\]

**模板只提供 `cover` 一个标量**，其余全是本片几何。这就是为什么模板可迁移——`cover` 是"人该占画面多高"，纯比例，换谁来都成立。

---

## 第 1 步：基准视窗 \(H_{\text{base}}\)（画幅几何，与内容无关）

\(z=1\) 时在源画面里裁多大。要求裁剪窗的宽高比 = 成片宽高比，且尽量大：

\[
\frac{W_s}{H_s} \ge \frac{W_o}{H_o} \;\Rightarrow\;
H_{\text{base}} = H_s,\quad W_{\text{base}} = H_s \cdot \frac{W_o}{H_o}
\]

> 本例：源 16:9、成片也 16:9 → `base = 1920 × 1080`，即整幅。

代码：`transform.largest_source_view()`

---

## 第 2 步：`cover` —— 模板唯一贡献的量（★学习环节在这里）

对参考视频**逐帧**做：

1. 选出该帧的 C位（全局 DP）
2. 判该帧景别（`_shot_from_exposure`，靠"被画框裁掉多少"）
3. 量 \(\mathrm{cover} = \dfrac{\text{C位 bbox 高}}{\text{画面高}}\)

再按景别取**中位数**（抗离群）：

\[
\mathrm{cover}[s] = \mathrm{median}\Big\{\tfrac{h_{\text{bbox}}(t)}{H_{\text{frame}}} \;:\; \mathrm{shot}(t)=s\Big\}
\]

实测某参考片学出来：

| 景别 | cover | 含义 |
| --- | --- | --- |
| `extreme_wide` | 0.40 | 人只占画面高的 40% |
| `wide` | 0.726 | 人占 73% |
| `medium` | 0.803 | 人（可见部分）占 80% |
| `closeup` | 1.0 | 规格反解，见下 |

同时还学 `cx_frac`（人放在画面横向哪儿）和 `head_top_frac`（头顶留白）。

代码：`learn_template._learn_shot_framing()`

> **为什么这个可迁移、而"C位是谁"不可迁移**：参考视频机位是**动**的，画面中央的人是摄影师主动**放**在那儿的；目标视频机位是**静**的，人是自己**站**在那儿的。两者没有身份对应关系。但"远景时人占画面 73%"是纯几何，换谁都成立。

---

## 第 3 步：\(h_{\text{content}}\) —— 这一帧要装进画面的内容有多高（逐帧，按 C位 骨架）

不同景别取身体的不同段。**颅顶**用解剖比例估（COCO-17 最高点是眉眼，不是头顶）：

\[
y_{\text{crown}} \approx y_{\text{nose}} - 0.65\,(y_{\text{shoulder}} - y_{\text{nose}})
\]

| 景别 | \(h_{\text{content}}\) | 规格 |
| --- | --- | --- |
| `wide` / `extreme_wide` | bbox 全高 | 全身 |
| `medium` | 颅顶 → 髋 + 0.6×躯干 | 到大腿中部 |
| `closeup` | \(2\,(y_{\text{hip}} - y_{\text{shoulder}})\) | **腰在下边框、肩在水平中线** |

`closeup` 是**从规格反解**的，不靠经验系数：

> 肩在水平中线 ⟹ 视窗中心 \(c_y = y_{\text{shoulder}}\)
> 髋在下边框 ⟹ \(\frac{H_v}{2} = y_{\text{hip}} - c_y\) ⟹ \(H_v = 2(y_{\text{hip}} - y_{\text{shoulder}})\)
> 头顶留白因此**自动**得到 ≈ 19% 画面高，不需要另外调。

所以 `closeup` 的 `cover` 必须 = 1.0——视窗高已经精确反解出来了，再乘系数就破坏规格。

代码：`camera._shot_targets()` / `camera._crown_y()`

---

## 第 4 步：合成 z

\[
\mathrm{cover} = \frac{h_{\text{content}}}{H_v},\qquad H_v = \frac{H_{\text{base}}}{z}
\quad\Longrightarrow\quad
z = \mathrm{cover}\cdot\frac{H_{\text{base}}}{h_{\text{content}}}
\]

本例（舞者身高 350px，颅顶 y=397 / 肩 y=463 / 髋 y=564 / 踝 y=750）：

| 景别 | \(h_{\text{content}}\) | 计算 | **z** |
| --- | --- | --- | --- |
| `extreme_wide` | 327.2 | 0.400 × 1080 / 327.2 | **1.32** |
| `wide` | 327.2 | 0.726 × 1080 / 327.2 | **2.40** |
| `medium` | 228.8 | 0.803 × 1080 / 228.8 | **3.79** |
| `closeup` | 203.0 | 1.000 × 1080 / 203.0 | **5.32** |

**直觉检查**：人在源画面里越小 → \(h_{\text{content}}\) 越小 → z 越大。景别越近 → \(h_{\text{content}}\) 越小（只取上半身）→ z 越大。两者都对。

---

## 第 5 步：两道夹取（这里决定了"设计想要"能不能实现）

### 5.1 安全框 \(z_{\text{safe}}\)（构图护栏）

裁剪窗必须能装下"该景别必须保住的身体部位"：

\[
z \le \min\Big(\frac{W_{\text{base}}}{w_{\text{box}}},\; \frac{H_{\text{base}}}{h_{\text{box}}}\Big)
\]

按景别分层——**这一层曾经把全片压成远景**：

| 景别 | 必须装下 | 允许切掉 |
| --- | --- | --- |
| wide / extreme_wide | `full`（含膝踝） | — |
| medium | `upper`（髋以上） | 腿 |
| closeup | `core`（头+肩+上胸） | 胸以下 |

> 若无条件夹 \(z_{\text{full}}\)，等于宣布"任何景别都保证全身入画"——中景要切腿、特写要切到胸，几何上永远不可能发生。`core`（头胸）始终是硬约束，任何景别都不破。

### 5.2 画质预算 \(z_{\max}\)（采样定律，绕不过）

\[
z_{\max} =
\begin{cases}
\min\Big(z_{\text{cfg}},\; \min\big(\tfrac{W_{\text{base}}}{W_o}, \tfrac{H_{\text{base}}}{H_o}\big)\Big), & \texttt{allow\_upscale=false}\\[6pt]
z_{\text{cfg}}, & \texttt{allow\_upscale=true}
\end{cases}
\]

本例：画质预算 = min(1920/1280, 1080/720) = **1.50**。

- `allow_upscale=false` → \(z_{\max}=1.50\)。**上面算的 medium(3.79) / closeup(5.32) 全部实现不了**，会被夹到 1.50 → 全片看着都是远景。
- `allow_upscale=true, max_zoom=6` → \(z_{\max}=6.00\)，近景能出来，代价是上采样掉画质。

> 这就是日志里那两行诊断的由来：
> ```
> [诊断] 设计想要 zoom：中位=3.80 最大=7.19 | 上限 emax=6.00
> [诊断] 2% 帧想推近却被 max_zoom 夹住 · 13% 帧被安全框额外压低
> ```
> `sat_safe` 必须与 \(\min(z_{\text{want}}, z_{\max})\) 比，否则会把 emax 的账算到安全框头上。

代码：`transform.effective_max_zoom()` / `camera._apply_safe_frame()`

---

## 第 6 步：时间维度（平滑 / 硬切 / 拍点）

1. **One Euro 自适应低通**：\(f_c = f_{\min} + \beta|\dot{\hat{x}}|\)，慢动强滤、快动少延迟
2. **死区 + 限速**：目标在安全区内不动；单帧位移 ≤ 32px
3. **硬切**：标记 `cut` 的段首帧 **reset 滤波器 + \(c_{\text{prev}} \leftarrow c_{\text{tgt}}\)** → 死区/限速自然不触发 → 裁剪窗瞬间跳。不这样做，限速会把 500px 的构图跳变抹成 16 帧缓慢推移（表现为"全是移动运镜、一个硬切也没有"）
4. **拍点脉冲**：`beat_pulse` / `downbeat_punch` 叠加到 z，叠完再过一次安全框夹取

---

## 全链路

```
参考视频 ──逐帧──► C位(全局DP) ──► 景别判定(被裁多少) ──► cover 中位数 ──► 模板
                                                                          │
目标视频 ──逐帧──► C位(全局DP) ──► 骨架 ──► h_content ◄────────────────────┤
                                              │                            │
                                              ▼                            ▼
                                     z = cover × H_base / h_content
                                              │
                                              ▼
                                   min(z, z_safe, z_max)
                                              │
                                              ▼
                                  One Euro + 死区 + 限速 + 硬切
```

---

## 边界（诚实说明）

1. **裁剪相机不是真推轨**：z 变大只是裁得更小再放大，背景不会产生视差。真 dolly 复现不了。
2. **源分辨率不足时近景必糊**：1920 源里 350px 的人切特写需 z≈5.3，等于把 ~200px 拉到 720p。采样定律，拉不回。
3. **`cover` 是中位数**：它反映"这个风格通常把人拍多大"，不是"这一刻该多大"。后者是剪辑师的即时判断，推不出来。
4. **`medium` 的 cover 可能大于 `wide`**：这不是 bug。cover 量的是**可见部分**占画面的比例——中景只拍上半身，但那半身填满画面，所以 cover 反而更高。
