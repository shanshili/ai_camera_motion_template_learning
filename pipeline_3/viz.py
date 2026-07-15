"""
可视化校验
==========
plot_music   : 波形 + 拍点/强拍 + 能量曲线 + 段落
plot_pose    : 运动能量 + 姿态事件时间线 + 质心高度
plot_shotplan: 分镜时间线 + 拍点，检查换镜是否踩点
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_music(result, out_png):
    dbg = result.get("_debug", {})
    y = dbg.get("y"); sr = 22050; fps = result["fps"]
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    if y is not None:
        ty = np.arange(len(y)) / sr
        step = max(1, len(y) // 6000)
        ax[0].plot(ty[::step], y[::step], color="#9aa", lw=0.5)
    for b in result["beat_grid"]:
        ax[0].axvline(b["t"], color="#d44" if b["is_downbeat"] else "#2a8",
                      lw=1.6 if b["is_downbeat"] else 0.8, alpha=0.9)
    ax[0].set_title(f"BPM={result['bpm']} beats={len(result['beat_grid'])} (red=downbeat)")
    ec = np.array(result["energy_curve"]); tf = np.arange(len(ec)) / fps
    ax[1].plot(tf, ec, color="#c70", lw=1.2); ax[1].fill_between(tf, ec, color="#c70", alpha=0.15)
    for s in result["sections"]:
        ax[1].axvspan(s["start_f"] / fps, s["end_f"] / fps,
                      color={"low": "#cfe", "mid": "#fec", "high": "#fcc"}.get(s["label"], "#eee"),
                      alpha=0.4)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return out_png


def plot_pose(result, out_png):
    pf = result["pose_frames"]; fps = result["fps"]
    frames = np.array([p["frame"] for p in pf])
    me = np.array([p["motion_energy"] for p in pf])
    cy = np.array([p["centroid"][1] if p["centroid"] else np.nan for p in pf])
    t = frames / fps
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].plot(t, me, color="#36c", lw=1.2); ax[0].fill_between(t, me, color="#36c", alpha=0.15)
    colors = {"jump": "#d44", "spin": "#84c", "extension": "#2a8",
              "freeze": "#888", "level_change": "#c70"}
    seen = set()
    for ev in result["pose_events"]:
        c = colors.get(ev["type"], "#333")
        ax[0].axvline(ev["frame"] / fps, color=c, lw=1.4, alpha=0.9,
                      label=ev["type"] if ev["type"] not in seen else None)
        seen.add(ev["type"])
    ax[0].set_ylabel("motion energy"); ax[0].legend(loc="upper right", fontsize=8, ncol=5)
    ax[1].plot(t, cy, color="#555", lw=1.2); ax[1].invert_yaxis()
    ax[1].set_ylabel("centroid y (px)"); ax[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return out_png


def plot_shotplan(plan, music, fps, out_png):
    shots = ["wide", "medium", "upper", "closeup"]
    ymap = {s: i for i, s in enumerate(shots)}
    fig, ax = plt.subplots(figsize=(13, 4))
    colors = {"follow": "#39c", "static": "#888", "push_in": "#e63",
              "pull_out": "#2a8", "roll": "#84c", "orbit": "#c70", "recenter": "#aaa"}
    for seg in plan:
        s = seg["start_f"] / fps; e = seg["end_f"] / fps
        y = ymap.get(seg["shot"], 1)
        ax.barh(y, e - s, left=s, height=0.6,
                color=colors.get(seg["move"], "#ccc"), alpha=0.8, edgecolor="w")
        ax.text((s + e) / 2, y, seg["move"], ha="center", va="center", fontsize=7)
    for b in (music or {}).get("beat_grid", []):
        ax.axvline(b["frame"] / fps, color="#d44" if b["is_downbeat"] else "#bbb",
                   lw=1.2 if b["is_downbeat"] else 0.5, alpha=0.6, zorder=0)
    ax.set_yticks(range(len(shots))); ax.set_yticklabels(shots)
    ax.set_xlabel("time (s)"); ax.set_title("shot plan vs beats (red=downbeat)")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return out_png
