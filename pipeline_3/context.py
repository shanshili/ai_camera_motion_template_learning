"""
管线上下文 (PipelineContext)
============================
各阶段共享的数据对象。相机含 zoom/cx/cy/rot/blur/move 字段，
rot 用于旋转运镜（render 会真正应用它）。
"""

from dataclasses import dataclass, field
from typing import Optional, List

from .timeline import Timeline


@dataclass
class CameraParams:
    """单帧虚拟相机参数。"""
    zoom: float = 1.0
    cx: Optional[float] = None
    cy: Optional[float] = None
    rot: float = 0.0            # 旋转角（度），正为逆时针
    blur: float = 0.0           # 方向运动模糊强度 0..1
    move: str = "follow"        # 当前帧的运镜标签（调试/可视化用）
    shot: str = "medium"        # 当前帧景别（wide/medium/upper/closeup）


@dataclass
class PipelineContext:
    config: dict
    input_path: str
    output_path: str

    meta: Optional[dict] = None
    timeline: Optional[Timeline] = None
    camera_track: Optional[List[CameraParams]] = None

    extras: dict = field(default_factory=dict)
