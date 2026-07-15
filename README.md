# AI 运镜系统 · 工程

音乐 + 多人姿态驱动的虚拟运镜。两条链路共用一套感知底座：

- **在线生成**：用户视频 → 感知 → 主体选择(C位) → 分镜(读模板) → 相机(安全执行) → 渲染
- **离线学习**：参考 MV → 反解可靠信号 → 产出 `template.json`（生成端读取）

两条总原则：**智能在上游**（选主体 / 分镜 / 模板），**安全在下游**（安全框 / 降级 / 夹取永远兜底）；一切量用**归一化相对量**（秒 / 画面比例 / 覆盖率 / 角色），绝对像素只在执行那一刻翻译。

---

## 目录结构

```
ai_camera_project/
├─ run.py                     # 在线生成入口
├─ config/default.yaml        # 配置（含 template.path / subject 权重 / 安全框）
├─ templates/                 # 学到的模板 json 放这里
├─ pipeline_3/                # 主管线包
│  ├─ runner.py stage.py context.py timeline.py   # 编排/契约/时间轴
│  ├─ ffio.py transform.py skeleton.py            # 编解码/相机变换/骨架
│  ├─ music.py pose_events.py                     # 音乐分析 / 姿态事件
│  ├─ yolo_pose.py            # 姿态纯助手（中心/插值/打包）
│  ├─ subject.py    ★新增     # 主体选择：C位角色化 + FEATURE/GROUP
│  ├─ template.py   ★新增     # 模板 IR 契约 + 默认模板 + 套用翻译
│  ├─ learn_template.py ★新增 # 离线学习脚本（独立运行）
│  ├─ viz.py                  # 可视化校验
│  └─ stages/
│     ├─ probe.py             # 探测/时间轴
│     ├─ analysis.py ★重写    # 多人 records → subject 选 C位 → pose/music
│     ├─ shotplan.py ★重写    # 模板驱动分镜（替换硬编码策略）
│     ├─ camera.py            # 相机执行（安全核，原版未动）
│     └─ render.py            # 三路输出 + 能量时间轴
├─ multi_person_tracking/     # 多人骨架前端（独立，需 ultralytics + 权重）
│  ├─ run_tracking.py         # YOLO-pose + ByteTrack → tracked_keypoints.json
│  └─ trackers/registry.py + configs/*.yaml
├─ patches/                   # 可选增强（构图留白补丁）
├─ tools/                     # selftest / track_stats 诊断
└─ requirements.txt
```

---

## 运行方式

### 前置
- `ffmpeg` / `ffprobe` 在 PATH 中。
- `pip install -r requirements.txt`。
- 多人骨架前端另需 `ultralytics` 与 YOLO-pose 权重（GPU 推荐）。

### A. 先产多人骨架（离线，每个视频一次）
```bash
python -m multi_person_tracking.run_tracking --source dataset/12-motion.mp4 --model weights/yolo26s-pose.pt --output-dir analysis_out/12-motion --tracker bytetrack_loose
python -m multi_person_tracking.run_tracking --source dataset/12.mp4 --model weights/yolo26s-pose.pt --output-dir analysis_out/12 --tracker bytetrack_loose

# 产出 analysis_out/demo/tracked_keypoints.json
```

### B. 在线生成运镜
```bash
python run.py --input dataset/12-motion.mp4 --output output/12-motion-1.mp4
python run.py --input dataset/12.mp4 --output output/12-1.mp4
# analysis 阶段以 reuse_existing 复用上一步的 tracked_keypoints.json，
# 并自行跑音乐分析、选 C位、出分镜、执行相机、渲染。
```
`config/default.yaml` 里 `template.path` 为空时用**内置默认模板**（已编码基本标准），无需先学模板即可跑。

### C. 学习一个风格模板（可选）
```bash
# 先对参考 MV 跑一次 A（产 tracked_keypoints.json）与音乐分析（跑一次 run.py 即会产 music.json）
python -m pipeline_3.learn_template --ref-video refs/12-motion.mp4 --analysis-dir analysis_out/12-motion --out templates/12-motion_v1.json --name 12-motion_v1
# 然后把 config 的 template.path 指向该 json，再跑 B 即用学到的风格。
```

---

## 文件归类（相对旧单人版）

| 归类 | 文件 |
| --- | --- |
| ★新增 | `pipeline_3/subject.py`、`template.py`、`learn_template.py` |
| ★重写 | `pipeline_3/stages/analysis.py`（多人）、`stages/shotplan.py`（模板驱动） |
| 小改 | `pipeline_3/music.py`（+abs_loudness）、`yolo_pose.py`（精简为纯助手）、`registry.py`（配置目录） |
| 原样复用 | `camera.py`（安全核）、`render.py`、`transform.py`、`ffio.py`、`timeline.py`、`context.py`、`stage.py`、`skeleton.py`、`pose_events.py`、`probe.py` |
| 独立前端 | `multi_person_tracking/*`（多人 YOLO + ByteTrack） |
| 可选增强 | `patches/camera_shot_targets_composition.py`（头顶留白/取景段/extreme_wide） |

---

## 基本标准落点（你的规格 → 代码）

- 节奏强→合重拍：`shotplan` 换镜吸附强拍 + 模板 `accent.downbeat_punch`。
- 动作大→跟随：模板 `event_map` jump/big_move→extreme_wide+pull_out。
- 舒缓→人占比多 / 活跃→稍多：`section_default` + `music.abs_loudness` 判安静歌全局收敛。
- 单人 特写/中景/远景 + 安全框：`camera` 安全核 + `_shot_targets`（留白见构图补丁）。
- 多人跟 C位：`subject`（FEATURE）；群体框 GROUP 掩码已产出，`shotplan` 对 GROUP 帧偏 wide。
- 姿态→景别、旋转优先级高：模板 `event_map`（含 spin→roll 高优先，修了旧版漏接）。
- 景别留白（头顶1/4、脚下1/6，手臂出镜）：模板 `shot_coverage` + 构图补丁生效。

---

## 诚实边界（避免验收当 bug）

1. 单源裁剪无法露出画外内容：真实推轨/摇臂/视差复现不了，只逼近构图意图。
2. 学习模块只反解**可复现信号**（景别分布 / 换镜节奏 / 卡点命中 / 段落倾向）；
   不做 homography 反解真实相机运动。单参考主要靠 `style_descriptor` + 默认模板承载风格。
3. GROUP 群体大远景：`shotplan` 已对 GROUP 帧偏 wide；若要 camera 用群体并集框做安全约束，
   需给 `camera._subject_geometry` 增加 group_box 路径（未包含，属下一步）。
4. 多因素仲裁（§5）尚未替换 camera 的加法脉冲；当前强调层是原版 beat_pulse/downbeat_punch。
5. 源分辨率不足时的近景会上采样掉画质（采样定律，无法恢复）。

详细设计与审计见随附的 `cinematography_system_doc.md`（另一份文档）。

## 模板预览（并入学习流程，非独立脚本）

学模板时会**自动**渲一段该模板的火柴人运镜预览（底部带动作时间轴），直观看学到的模板怎么运镜：

```bash
python -m pipeline_3.learn_template --ref-video refs/mv.mp4 --analysis-dir analysis_out/mv \
    --out templates/mv_v1.json --name mv_v1
# 产出 templates/mv_v1.json 与 templates/mv_v1_preview.mp4（按模板朝向渲染）
# 不需要预览：加 --no-preview
```

## 模板的朝向/人数/节奏限定（套用前判别）

学出的模板 meta 记录：**朝向**（landscape/portrait/square）、**出镜人数区间**（min/max/median）、**BPM**（+容差）。
套用到别的视频时，`shotplan` 会算兼容分并打印建议：
- 硬否决（compat=0）：朝向不一致（横向模板套竖屏）、群舞模板套独舞、人数差太多。
- 软距离：BPM 偏差、人数中位偏差。
- 分数 <0.3 给告警，但不硬拦——决定权在你，可换模板或用内置默认模板。
