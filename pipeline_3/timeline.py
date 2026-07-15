"""
统一时间轴 (Timeline)
=====================
整条管线只认这一个时间基准：帧索引 <-> 时间戳的换算全部走这里，
所有模块（音乐拍点、姿态事件帧号、相机关键帧）都对齐到它，
这是"时间轴对齐无漂移"的根本保证。
"""

from dataclasses import dataclass
from fractions import Fraction


@dataclass
class Timeline:
    fps: Fraction          # 用分数表示，精确支持 60000/1001 这类非整数帧率
    frame_count: int       # 总帧数（唯一真值）

    @property
    def duration(self) -> float:
        return float(self.frame_count / self.fps)

    def time_of(self, frame_index: int) -> float:
        return float(frame_index / self.fps)

    def frame_of(self, t: float) -> int:
        return int(round(t * float(self.fps)))

    def assert_track_length(self, n: int) -> None:
        if n != self.frame_count:
            raise ValueError(
                f"轨迹长度 {n} 与总帧数 {self.frame_count} 不一致——会导致漂移"
            )

    def __str__(self) -> str:
        return (f"Timeline(fps={float(self.fps):.3f}, "
                f"frames={self.frame_count}, duration={self.duration:.3f}s)")
